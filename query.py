import os
import sys
import time
from typing import List, Dict, Any

from dotenv import load_dotenv
from pinecone import Pinecone
import google.generativeai as genai


INDEX_NAME = "adgm-compliance"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GEMINI_EMBED_MODEL = "text-embedding-004"      # 768-dim


def _load_env_and_clients() -> Dict[str, Any]:
    load_dotenv()

    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise RuntimeError("Missing PINECONE_API_KEY in .env")
    pc = Pinecone(api_key=pinecone_api_key)

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise RuntimeError("Missing GEMINI_API_KEY in .env")
    genai.configure(api_key=gemini_api_key)

    return {"pc": pc}


def _get_index_and_dim(pc: Pinecone):
    idx = pc.Index(INDEX_NAME)
    # Try to fetch index description to infer dimension
    try:
        desc = pc.describe_index(INDEX_NAME)
        # dimension may be at top-level or inside spec
        dim = (
            (desc.get("dimension") if isinstance(desc, dict) else None)
            or (desc.get("spec", {}).get("pod", {}).get("dimension") if isinstance(desc, dict) else None)
            or 768  # default fallback used in ingestion
        )
    except Exception:
        dim = 768
    return idx, int(dim)


def _embed_query(query: str, target_dim: int) -> List[float]:
    # Gemini-only embedding, expect 768-dim index
    if target_dim != 768:
        raise RuntimeError(f"Index dimension {target_dim} not supported with Gemini embeddings (expected 768)")
    # Retry wrapper for rate limits
    delay = 2
    for attempt in range(4):
        try:
            res = genai.embed_content(model=GEMINI_EMBED_MODEL, content=query)
            break
        except Exception as e:
            if attempt == 3:
                raise
            time.sleep(delay)
            delay = min(60, delay * 2)
    emb = res.get("embedding") if isinstance(res, dict) else getattr(res, "embedding", None)
    if isinstance(emb, dict):
        emb = emb.get("values")
    if emb is None:
        raise RuntimeError("Failed to create Gemini query embedding")
    return list(emb)


def search_pinecone(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    clients = _load_env_and_clients()
    pc: Pinecone = clients["pc"]

    pc_index, dim = _get_index_and_dim(pc)
    query_vec = _embed_query(query, dim)

    res = pc_index.query(vector=query_vec, top_k=top_k, include_values=False, include_metadata=True)

    # Normalize result structure
    matches = []
    for m in (res.get("matches", []) if isinstance(res, dict) else getattr(res, "matches", [])):
        metadata = m.get("metadata", {}) if isinstance(m, dict) else getattr(m, "metadata", {})
        score = m.get("score") if isinstance(m, dict) else getattr(m, "score", None)
        text = metadata.get("text") or ""
        matches.append({"text": text, "score": score, "metadata": metadata})
    return matches


def _build_context(chunks: List[Dict[str, Any]], max_chars: int = 12000) -> str:
    parts: List[str] = []
    total = 0
    for i, ch in enumerate(chunks):
        txt = (ch.get("text") or ch["metadata"].get("text") or "").strip()
        if not txt:
            continue
        block = f"[Chunk {i+1}]\n{txt}\n"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n".join(parts).strip()


def answer_with_gemini(query: str) -> str:
    matches = search_pinecone(query, top_k=5)
    context = _build_context(matches)

    if not context:
        return "I could not find an answer in the provided documents."

    system_instruction = (
        "You are an ADGM compliance assistant. Answer the user's question strictly "
        "using ONLY the provided context. If the answer is not clearly supported "
        "by the context, respond exactly with: \"I could not find an answer in the provided documents.\""
    )

    prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer concisely and cite relevant points from the context."
    )

    # Retry wrapper for generation
    delay = 2
    for attempt in range(4):
        try:
            model = genai.GenerativeModel(GEMINI_MODEL)
            resp = model.generate_content(prompt)
            break
        except Exception:
            if attempt == 3:
                raise
            time.sleep(delay)
            delay = min(60, delay * 2)

    # Extract text safely
    answer = getattr(resp, "text", None)
    if not answer and hasattr(resp, "candidates") and resp.candidates:
        try:
            parts = resp.candidates[0].content.parts
            answer = "".join(getattr(p, "text", "") for p in parts)
        except Exception:
            answer = None

    if not answer:
        return "I could not find an answer in the provided documents."

    answer = answer.strip()
    if not answer:
        return "I could not find an answer in the provided documents."

    return answer


if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:]).strip()
    else:
        user_query = input("Enter your question: ").strip()

    if not user_query:
        print("Please provide a non-empty question.")
        sys.exit(1)

    try:
        answer = answer_with_gemini(user_query)
        print(answer)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

