import os
import sys
import time
import hashlib
from typing import Iterable, List, Tuple

from dotenv import load_dotenv


# Optional dependencies loaded lazily with clear errors
try:
    import fitz  # PyMuPDF for PDFs
except Exception:  # pragma: no cover - optional at import time
    fitz = None  # type: ignore

try:
    import docx  # python-docx for DOCX
except Exception:  # pragma: no cover
    docx = None  # type: ignore

try:
    import tiktoken  # tokenization for chunking
except Exception:  # pragma: no cover
    tiktoken = None  # type: ignore

try:
    import google.generativeai as genai  # Gemini API
except Exception:
    print("ERROR: google-generativeai package not available. Install with: pip install google-generativeai", file=sys.stderr)
    raise

try:
    from pinecone import Pinecone, ServerlessSpec  # New Pinecone SDK
except Exception:
    print("ERROR: pinecone package not available. Install with: pip install pinecone", file=sys.stderr)
    raise


# ==============================
# Configuration (easy to change)
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "adgm_raw")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "adgm-compliance")

# Embeddings / chunking (Gemini)
GEMINI_EMBEDDING_MODEL = "text-embedding-004"  # 768-dim
CHUNK_SIZE_TOKENS = 800
CHUNK_OVERLAP_TOKENS = 100
EMBED_BATCH_SIZE = 64
UPSERT_BATCH_SIZE = 100


def log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def load_env() -> None:
    load_dotenv()


def configure_gemini() -> None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY in environment/.env")
    genai.configure(api_key=api_key)


def get_pinecone_client() -> Pinecone:
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing PINECONE_API_KEY in environment/.env")
    return Pinecone(api_key=api_key)


def get_or_create_index(pc: Pinecone, name: str, dimension: int) -> "Index":
    indexes = [idx["name"] for idx in (pc.list_indexes() or [])]
    if name not in indexes:
        cloud = os.getenv("PINECONE_CLOUD", "aws")
        region = os.getenv("PINECONE_REGION", "us-east-1")
        log(f"Creating Pinecone index '{name}' (dim={dimension}, metric=cosine, cloud={cloud}, region={region})...")
        pc.create_index(
            name=name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
        # Wait for index to be ready
        while True:
            info = pc.describe_index(name)
            if info and info.get("status", {}).get("ready"):
                break
            time.sleep(2)
    return pc.Index(name)


def read_text_from_pdf(path: str) -> str:
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) is required to read PDFs. Install with: pip install pymupdf")
    text_parts: List[str] = []
    with fitz.open(path) as doc:
        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            text_parts.append(page.get_text("text") or "")
    return "\n".join(text_parts)


def read_text_from_docx(path: str) -> str:
    if docx is None:
        raise RuntimeError("python-docx is required to read .docx. Install with: pip install python-docx")
    document = docx.Document(path)
    paragraphs = [p.text for p in document.paragraphs if p.text]
    return "\n".join(paragraphs)


def read_text_from_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def get_encoder():
    if tiktoken is None:
        log("WARN: tiktoken not installed; using whitespace-based tokenization as fallback.")
        return None
    try:
        # cl100k_base is a good approximation for chunk sizing
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def tokenize(text: str, encoder) -> List[int]:
    if encoder is None:
        # Fallback: approximate tokens by splitting on whitespace
        # Represent tokens as indices for chunk math only; actual IDs are not used
        return list(range(len(text.split())))
    return encoder.encode(text)


def detokenize(tokens: List[int], encoder, original_text: str) -> str:
    if encoder is None:
        # Fallback: reconstruct from whitespace token count; not perfect, but close
        words = original_text.split()
        # tokens are indices into a counted list; slice by token count
        count = len(tokens)
        return " ".join(words[:count])
    return encoder.decode(tokens)


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    encoder = get_encoder()
    token_ids = tokenize(text, encoder)
    chunks: List[str] = []
    if not token_ids:
        return chunks

    step = max(1, chunk_size - overlap)
    total = len(token_ids)
    for start in range(0, total, step):
        end = min(start + chunk_size, total)
        window = token_ids[start:end]
        if not window:
            continue
        # For encoder=None fallback, detokenize using original text approximation
        if encoder is None:
            # We cannot precisely map back tokens to text; approximate by splitting words
            words = text.split()
            chunk_words = words[start:end] if end <= len(words) else words[start:]
            chunk = " ".join(chunk_words)
        else:
            chunk = detokenize(window, encoder, text)
        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)
        if end >= total:
            break
    return chunks


def list_input_files(input_dir: str) -> List[str]:
    supported_exts = {".pdf", ".docx", ".txt"}
    files: List[str] = []
    for entry in os.listdir(input_dir):
        path = os.path.join(input_dir, entry)
        if os.path.isfile(path) and os.path.splitext(entry)[1].lower() in supported_exts:
            files.append(path)
    return sorted(files)


def guess_source_url(file_path: str) -> str:
    # Sidecar pattern: filename.ext.url containing the original URL
    sidecar = file_path + ".url"
    if os.path.exists(sidecar):
        try:
            with open(sidecar, "r", encoding="utf-8") as f:
                url = f.read().strip()
                return url
        except Exception:
            return ""
    return ""


def sha1_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def batched(iterable: List, batch_size: int) -> Iterable[List]:
    for i in range(0, len(iterable), batch_size):
        yield iterable[i : i + batch_size]


def embed_texts(texts: List[str]) -> List[List[float]]:
    vectors: List[List[float]] = []
    for text in texts:
        # Gemini embedding (text-embedding-004)
        res = genai.embed_content(model=GEMINI_EMBEDDING_MODEL, content=text)
        # Response may be dict-like; prefer res["embedding"] if present
        emb = None
        if isinstance(res, dict):
            emb = res.get("embedding") or res.get("values")
        else:
            # Some versions return objects with .embedding
            emb = getattr(res, "embedding", None)
            if isinstance(emb, dict):
                emb = emb.get("values")
        if emb is None:
            raise RuntimeError("Failed to parse embedding response from Gemini API")
        vectors.append(list(emb))
    return vectors


def process_and_upload_file(
    file_path: str,
    index,
) -> Tuple[int, int]:
    """Returns (num_chunks, num_upserted)."""
    filename = os.path.basename(file_path)
    ext = os.path.splitext(filename)[1].lower()

    try:
        if ext == ".pdf":
            text = read_text_from_pdf(file_path)
        elif ext == ".docx":
            text = read_text_from_docx(file_path)
        elif ext == ".txt":
            text = read_text_from_txt(file_path)
        else:
            log(f"SKIP: Unsupported file type: {filename}")
            return 0, 0
    except Exception as err:
        log(f"ERROR: Failed to read {filename}: {err}")
        return 0, 0

    text = text.strip()
    if not text:
        log(f"SKIP: Empty content in {filename}")
        return 0, 0

    chunks = chunk_text(text, CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS)
    if not chunks:
        log(f"SKIP: No chunks produced for {filename}")
        return 0, 0

    source_url = guess_source_url(file_path)

    total_upserted = 0
    # Embed in batches
    for chunk_batch_indices in batched(list(range(len(chunks))), EMBED_BATCH_SIZE):
        batch_texts = [chunks[i] for i in chunk_batch_indices]
        try:
            embeddings = embed_texts(batch_texts)
        except Exception as err:
            log(f"ERROR: Embedding generation failed for {filename} batch: {err}")
            continue

        # Assemble vectors with metadata
        vectors = []
        for local_idx, embedding in zip(chunk_batch_indices, embeddings):
            chunk_index = int(local_idx)
            uid = sha1_id(f"{file_path}::{chunk_index}")
            metadata = {
                "source": filename,
                "url": source_url,
                "chunk_index": chunk_index,
                "text": chunks[chunk_index],
            }
            vectors.append((uid, embedding, metadata))

        # Upsert in Pinecone in batches
        for upsert_batch in batched(vectors, UPSERT_BATCH_SIZE):
            try:
                index.upsert(vectors=upsert_batch)
                total_upserted += len(upsert_batch)
            except Exception as err:
                log(f"ERROR: Pinecone upsert failed for {filename}: {err}")

    return len(chunks), total_upserted


def main() -> None:
    load_env()

    if not os.path.isdir(INPUT_DIR):
        raise RuntimeError(f"Input directory not found: {INPUT_DIR}")

    log("Configuring Gemini client...")
    configure_gemini()

    log("Initializing Pinecone client...")
    pc = get_pinecone_client()

    # Determine embedding dimension for index creation (Gemini text-embedding-004 => 768)
    embedding_dim = 768
    index = get_or_create_index(pc, INDEX_NAME, embedding_dim)

    files = list_input_files(INPUT_DIR)
    if not files:
        log(f"No supported files found in {INPUT_DIR}")
        return

    log(f"Found {len(files)} file(s) to process.")

    total_chunks = 0
    total_upserted = 0
    for file_path in files:
        filename = os.path.basename(file_path)
        log(f"Processing: {filename}")
        num_chunks, num_up = process_and_upload_file(file_path, index)
        total_chunks += num_chunks
        total_upserted += num_up
        log(f"Done: {filename} | chunks={num_chunks} upserted={num_up}")

    log(f"All done. Total chunks: {total_chunks}, Total upserted: {total_upserted}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Interrupted by user.")
        try:
            sys.exit(1)
        except SystemExit:
            os._exit(1)
