import os
import re
import sys
import time
from typing import List, Set, Tuple
from urllib.parse import urlparse

import fitz  # PyMuPDF
import requests
from bs4 import BeautifulSoup


# ==============================
# Configuration (easy to change)
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_DIR, "Data Sources.pdf")
OUTPUT_DIR = os.path.join(BASE_DIR, "adgm_raw")

# Networking defaults
REQUEST_TIMEOUT_SECONDS = 30
HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    )
}


def log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def ensure_output_dir(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)


def sanitize_text(text: str) -> str:
    # Collapse excessive blank lines and strip
    lines = [line.strip() for line in text.splitlines()]
    # Remove empty lines at ends and collapse multiples
    cleaned_lines: List[str] = []
    previous_blank = False
    for line in lines:
        if line:
            cleaned_lines.append(line)
            previous_blank = False
        else:
            if not previous_blank:
                cleaned_lines.append("")
                previous_blank = True
    return "\n".join(cleaned_lines).strip()


def sanitize_filename(name: str) -> str:
    # Replace non-alphanumeric with underscores; preserve dots (for extensions)
    name = re.sub(r"[^A-Za-z0-9._]+", "_", name)
    # Collapse repeated underscores
    name = re.sub(r"_+", "_", name)
    # Avoid leading/trailing underscores or dots
    name = name.strip("._")
    # Limit length to avoid filesystem issues
    if len(name) > 200:
        root, ext = os.path.splitext(name)
        name = root[:200 - len(ext)] + ext
    # Fallback
    return name or "file"


def unique_path(output_dir: str, filename: str) -> str:
    """Return a unique filepath inside output_dir by appending numeric suffixes if needed."""
    base, ext = os.path.splitext(filename)
    candidate = os.path.join(output_dir, filename)
    counter = 1
    while os.path.exists(candidate):
        candidate = os.path.join(output_dir, f"{base}_{counter}{ext}")
        counter += 1
    return candidate


def safe_filename_from_url(url: str, default_ext: str = "") -> str:
    parsed = urlparse(url)
    path = parsed.path or ""
    name = os.path.basename(path)
    if not name:
        # Use host and path as a base if no basename present
        host_part = parsed.netloc.replace(":", "_")
        path_part = path.replace("/", "_").strip("_")
        combined = f"{host_part}_{path_part}" if path_part else host_part or "download"
        name = combined
    base, ext = os.path.splitext(name)
    if not ext and default_ext:
        ext = default_ext
    safe = sanitize_filename(base) + ext
    return safe


def extract_urls(pdf_path: str) -> List[str]:
    """Extract URLs from both explicit link annotations and visible text."""
    urls: Set[str] = set()
    url_like_pattern = re.compile(r"https?://[^\s)\]>\[\"'<>]+", re.IGNORECASE)

    if not os.path.exists(pdf_path):
        log(f"ERROR: PDF not found at path: {pdf_path}")
        return []

    try:
        with fitz.open(pdf_path) as doc:
            for page_index in range(doc.page_count):
                page = doc.load_page(page_index)

                # 1) Explicit link annotations
                try:
                    for link in page.get_links():
                        uri = link.get("uri")
                        if isinstance(uri, str) and uri.lower().startswith(("http://", "https://")):
                            urls.add(uri.strip())
                except Exception as err:  # PyMuPDF versions may vary
                    log(f"WARN: Failed to read link annotations on page {page_index + 1}: {err}")

                # 2) URL-looking text
                try:
                    text = page.get_text("text") or ""
                    for match in url_like_pattern.findall(text):
                        # Trim trailing punctuation that commonly clings to URLs
                        cleaned = match.rstrip(".,);]\")'")
                        urls.add(cleaned)
                except Exception as err:
                    log(f"WARN: Failed to extract text on page {page_index + 1}: {err}")
    except Exception as err:
        log(f"ERROR: Unable to open PDF: {err}")
        return []

    extracted = sorted(urls)
    log(f"Found {len(extracted)} unique URLs in PDF")
    return extracted


def separate_links(urls: List[str]) -> Tuple[List[str], List[str]]:
    direct_exts = {".pdf", ".docx", ".doc"}
    direct_links: List[str] = []
    webpage_links: List[str] = []

    for url in urls:
        try:
            parsed = urlparse(url)
            path = parsed.path or ""
            _, ext = os.path.splitext(path.lower())
            if ext in direct_exts:
                direct_links.append(url)
            else:
                webpage_links.append(url)
        except Exception:
            # Malformed URLs are skipped
            log(f"WARN: Skipping malformed URL: {url}")
            continue

    log(f"Separated into {len(direct_links)} file links and {len(webpage_links)} webpage links")
    return direct_links, webpage_links


def download_files(links: List[str], output_dir: str) -> None:
    ensure_output_dir(output_dir)

    for url in links:
        try:
            filename = safe_filename_from_url(url)
            filepath = unique_path(output_dir, filename)
            log(f"Downloading file: {url} -> {os.path.basename(filepath)}")

            with requests.get(url, headers=HTTP_HEADERS, timeout=REQUEST_TIMEOUT_SECONDS, stream=True) as resp:
                resp.raise_for_status()
                with open(filepath, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:  # filter out keep-alive chunks
                            f.write(chunk)
            log(f"SUCCESS: Saved file to {filepath}")
        except requests.RequestException as err:
            log(f"ERROR: Failed to download {url}: {err}")
        except Exception as err:
            log(f"ERROR: Unexpected error downloading {url}: {err}")


def _remove_noise_elements(soup: BeautifulSoup) -> None:
    # Remove scripts, styles, and other non-content elements
    for tag_name in [
        "script",
        "style",
        "noscript",
        "header",
        "footer",
        "nav",
        "aside",
        "form",
        "svg",
        "canvas",
    ]:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Remove common menu/header/footer/sidebar-like elements by class/id heuristics
    noise_keywords = [
        "menu",
        "navbar",
        "header",
        "footer",
        "breadcrumb",
        "sidebar",
        "cookie",
        "consent",
        "subscribe",
        "promo",
        "ad-",
        "advert",
        "social",
    ]
    selectors: List[str] = []
    for word in noise_keywords:
        selectors.append(f"*[class*='{word}']")
        selectors.append(f"*[id*='{word}']")
        selectors.append(f"*[role='{word}']")

    for elem in soup.select(", ".join(selectors)):
        try:
            elem.decompose()
        except Exception:
            continue


def scrape_webpages(links: List[str], output_dir: str) -> None:
    ensure_output_dir(output_dir)

    for url in links:
        try:
            filename = safe_filename_from_url(url, default_ext=".txt")
            if not filename.lower().endswith(".txt"):
                # Ensure .txt extension for webpage scrape outputs
                base, _ = os.path.splitext(filename)
                filename = base + ".txt"
            filepath = unique_path(output_dir, filename)
            log(f"Scraping page: {url} -> {os.path.basename(filepath)}")

            resp = requests.get(url, headers=HTTP_HEADERS, timeout=REQUEST_TIMEOUT_SECONDS)
            resp.raise_for_status()

            content_type = (resp.headers.get("Content-Type") or "").lower()
            if "text/html" not in content_type and not resp.text.strip().startswith("<"):
                # Not HTML; save raw text best-effort
                text_data = resp.text
            else:
                soup = BeautifulSoup(resp.text, "html.parser")
                _remove_noise_elements(soup)
                text_data = soup.get_text(separator="\n")

            cleaned = sanitize_text(text_data)

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(cleaned)
            log(f"SUCCESS: Saved text to {filepath}")
        except requests.RequestException as err:
            log(f"ERROR: Failed to scrape {url}: {err}")
        except Exception as err:
            log(f"ERROR: Unexpected error scraping {url}: {err}")


def main() -> None:
    log("Starting ingestion...")
    log(f"PDF path: {PDF_PATH}")
    log(f"Output directory: {OUTPUT_DIR}")
    ensure_output_dir(OUTPUT_DIR)

    urls = extract_urls(PDF_PATH)
    if not urls:
        log("No URLs found. Exiting.")
        return

    direct_links, webpage_links = separate_links(urls)

    if direct_links:
        download_files(direct_links, OUTPUT_DIR)
    else:
        log("No direct file links to download.")

    if webpage_links:
        scrape_webpages(webpage_links, OUTPUT_DIR)
    else:
        log("No webpage links to scrape.")

    log("Ingestion completed.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Interrupted by user.")
        try:
            sys.exit(1)
        except SystemExit:
            os._exit(1)

