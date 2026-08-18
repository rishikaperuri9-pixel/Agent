"""
Website crawler: BFS-crawls a domain up to a page/depth limit,
strips boilerplate (nav/footer/script/style), returns clean page text.
"""
import time
from collections import deque
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0 (RAG-Assessment-Bot/1.0)"}
BOILERPLATE_TAGS = ["nav", "footer", "header", "script", "style", "noscript", "svg", "form"]


def _same_domain(base: str, url: str) -> bool:
    return urlparse(base).netloc == urlparse(url).netloc


def _normalize(url: str) -> str:
    """Collapses trivial duplicates: trailing slash, query string, fragment."""
    parsed = urlparse(url)
    path = parsed.path.rstrip("/") or "/"
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def _clean_text(soup: BeautifulSoup) -> str:
    for tag in soup.find_all(BOILERPLATE_TAGS):
        tag.decompose()
    main = soup.find("main") or soup.find("article") or soup.body or soup
    text = main.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


def crawl(start_url: str, max_pages: int = 20, max_depth: int = 3, delay: float = 0.5):
    """
    Returns a list of dicts: {"url": str, "title": str, "text": str}
    """
    visited = set()
    queue = deque([(_normalize(start_url), 0)])
    pages = []

    while queue and len(pages) < max_pages:
        url, depth = queue.popleft()
        if url in visited or depth > max_depth:
            continue
        visited.add(url)

        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            resp.raise_for_status()
            if "text/html" not in resp.headers.get("Content-Type", ""):
                continue
        except requests.RequestException as e:
            print(f"  [skip] {url} ({e})")
            continue

        soup = BeautifulSoup(resp.text, "html.parser")
        title = soup.title.string.strip() if soup.title and soup.title.string else url
        text = _clean_text(soup)

        if len(text) > 200:  # skip near-empty pages
            pages.append({"url": url, "title": title, "text": text})
            print(f"  [{len(pages)}/{max_pages}] {url} ({len(text)} chars)")

        if depth < max_depth:
            for link in soup.find_all("a", href=True):
                next_url = _normalize(urljoin(url, link["href"]))
                if _same_domain(start_url, next_url) and next_url not in visited:
                    # skip common non-content links
                    if any(next_url.lower().endswith(ext) for ext in
                           [".pdf", ".jpg", ".png", ".zip", ".svg", ".css", ".js"]):
                        continue
                    queue.append((next_url, depth + 1))

        time.sleep(delay)  # be polite

    return pages


if __name__ == "__main__":
    import sys
    import json

    url = sys.argv[1] if len(sys.argv) > 1 else "https://example.com"
    results = crawl(url, max_pages=10)
    print(json.dumps(results, indent=2)[:1000])
