import time
import urllib.parse
import urllib.robotparser
from collections import deque

import requests
from bs4 import BeautifulSoup

import config
from storage.db import init_db, insert_page, get_page_count


_HEADERS = {"User-Agent": "SearchEngineBot/1.0 (educational project)"}


def _can_fetch(rp_cache: dict, url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    if origin not in rp_cache:
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(f"{origin}/robots.txt")
        try:
            rp.read()
        except Exception:
            rp_cache[origin] = None
            return True
        rp_cache[origin] = rp
    rp = rp_cache[origin]
    return rp is None or rp.can_fetch(_HEADERS["User-Agent"], url)


def _fetch(url: str) -> tuple[str, str, list[str]] | None:
    """Return (title, body_text, outbound_links) or None on error."""
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=10)
        resp.raise_for_status()
        if "text/html" not in resp.headers.get("Content-Type", ""):
            return None
    except Exception as e:
        print(f"  [skip] {url} — {e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    title = soup.title.string.strip() if soup.title and soup.title.string else url

    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    body = " ".join(soup.get_text(separator=" ").split())

    base = urllib.parse.urlparse(url)
    links: list[str] = []
    for a in soup.find_all("a", href=True):
        href = urllib.parse.urljoin(url, a["href"])
        parsed = urllib.parse.urlparse(href)
        # keep same-host http(s) links, strip fragments
        if parsed.scheme in ("http", "https") and parsed.netloc == base.netloc:
            clean = href.split("#")[0]
            if clean and clean not in links:
                links.append(clean)

    return title, body, links


def crawl(
    seed_urls: list[str] = config.SEED_URLS,
    max_depth: int = config.CRAWL_DEPTH,
    max_pages: int = config.MAX_PAGES,
    delay: float = config.CRAWL_DELAY,
) -> int:
    init_db()

    visited: set[str] = set()
    queue: deque[tuple[str, int]] = deque((url, 0) for url in seed_urls)
    rp_cache: dict = {}
    saved = 0

    while queue and saved < max_pages:
        url, depth = queue.popleft()
        if url in visited or depth > max_depth:
            continue
        visited.add(url)

        if not _can_fetch(rp_cache, url):
            print(f"  [robots] blocked: {url}")
            continue

        print(f"[{saved+1}] depth={depth} {url}")
        result = _fetch(url)
        if result is None:
            continue

        title, body, links = result
        doc_id = insert_page(url, title, body, links)
        if doc_id is not None:
            saved += 1

        if depth < max_depth:
            for link in links:
                if link not in visited:
                    queue.append((link, depth + 1))

        time.sleep(delay)

    print(f"\nCrawl complete. {saved} new pages saved (total in DB: {get_page_count()}).")
    return saved
