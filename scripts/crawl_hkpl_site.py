#!/usr/bin/env python3

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import hashlib
import logging
import re
from collections import deque
from urllib.parse import urljoin, urlparse, urldefrag

import requests
from bs4 import BeautifulSoup

from src.ingestion.registry import find_active_web_document_by_source_url
from src.ingestion.service import UPLOAD_DIR, ingest_path_sync

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hkpl_crawler")

SEED_URLS = [
    "https://www.hkpl.gov.hk/en/",
    "https://www.hkpl.gov.hk/en/extension-activities/",
    "https://www.hkpl.gov.hk/en/about-us/services/",
    "https://www.hkpl.gov.hk/en/e-resources/",
    "https://www.hkpl.gov.hk/en/collections/",
]

ALLOWED_HOST_SUFFIXES = (
    "hkpl.gov.hk",
    "lcsd.gov.hk",
)

MAX_PAGES = 30
MAX_DEPTH = 1
TIMEOUT = 30

CRAWL_STATE_DIR = Path("/app/storage/crawler_state")
CRAWL_STATE_DIR.mkdir(parents=True, exist_ok=True)


def normalize_url(url: str) -> str:
    url, _fragment = urldefrag(url)
    return url.strip()


def is_allowed_url(url: str) -> bool:
    parsed = urlparse(url)
    host = parsed.hostname.lower() if parsed.hostname else ""
    path = parsed.path.lower()

    if parsed.scheme not in {"http", "https"}:
        return False

    # Keep crawler focused on the main HKPL website.
    # This is domain-level control, not page-specific hardcoding.
    if host != "www.hkpl.gov.hk":
        return False

    blocked_ext = (
        ".jpg", ".jpeg", ".png", ".gif", ".svg", ".css", ".js",
        ".ico", ".zip", ".mp4", ".mp3", ".avi", ".mov",
    )

    if path.endswith(blocked_ext):
        return False

    return True


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def extract_title(soup: BeautifulSoup, url: str) -> str:
    if soup.title:
        return clean_text(soup.title.get_text(" ", strip=True))
    h1 = soup.find("h1")
    if h1:
        return clean_text(h1.get_text(" ", strip=True))
    return url


def extract_main_html(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup.select(
        "script, style, nav, footer, header, noscript, svg, "
        ".breadcrumb, .breadcrumbs, .side_menu, .sidebar, "
        ".share, .social, .pagination, .search, .menu"
    ):
        tag.decompose()

    title = extract_title(soup, "")

    candidates = [
        "main",
        "article",
        "#content",
        ".content",
        ".main_content",
        ".content_detail",
        "#main-content",
    ]

    best = None
    best_len = 0

    for selector in candidates:
        found = soup.select_one(selector)
        if found:
            text_len = len(clean_text(found.get_text(" ", strip=True)))
            if text_len > best_len:
                best = found
                best_len = text_len

    if best is None:
        best = soup.body or soup

    extracted_text = clean_text(best.get_text(" ", strip=True))

    # Generic content-quality filter.
    # This avoids indexing empty/login/search/navigation pages
    # without hardcoding specific URL paths.
    if len(extracted_text) < 200:
        raise ValueError(
            f"Skipped page because extracted text is too short: {len(extracted_text)} characters"
        )

    return title, str(best)


def page_hash(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    text = clean_text(soup.get_text(" ", strip=True))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def state_path_for_url(url: str) -> Path:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return CRAWL_STATE_DIR / f"{digest}.sha256"


def has_changed(url: str, content_hash: str) -> bool:
    state_path = state_path_for_url(url)
    old_hash = state_path.read_text().strip() if state_path.exists() else ""
    return old_hash != content_hash


def save_hash(url: str, content_hash: str) -> None:
    state_path_for_url(url).write_text(content_hash)


def save_html_for_ingestion(url: str, html: str) -> Path:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    path = UPLOAD_DIR / f"crawler_{digest}.html"
    path.write_text(html, encoding="utf-8")
    return path


def fetch(url: str) -> str:
    response = requests.get(
        url,
        timeout=TIMEOUT,
        headers={"User-Agent": "HKPL-RAG-Crawler/1.0"},
    )
    response.raise_for_status()
    return response.text


def discover_links(base_url: str, html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = []

    for tag in soup.find_all("a", href=True):
        absolute = normalize_url(urljoin(base_url, tag["href"]))
        if is_allowed_url(absolute):
            links.append(absolute)

    return sorted(set(links))


def crawl() -> dict:
    visited = set()
    queue = deque((url, 0) for url in SEED_URLS)

    stats = {
        "visited": 0,
        "indexed": 0,
        "unchanged": 0,
        "failed": 0,
        "discovered": 0,
    }

    while queue and stats["visited"] < MAX_PAGES:
        url, depth = queue.popleft()
        url = normalize_url(url)

        if url in visited:
            continue

        if not is_allowed_url(url):
            continue

        visited.add(url)
        stats["visited"] += 1

        try:
            logger.info("Fetching depth=%s url=%s", depth, url)
            raw_html = fetch(url)
            title, main_html = extract_main_html(raw_html)
            content_hash = page_hash(main_html)

            if has_changed(url, content_hash):
                path = save_html_for_ingestion(url, main_html)
                existing_document = find_active_web_document_by_source_url(url)
                previous_path = (
                    UPLOAD_DIR / existing_document["stored_file_name"]
                    if existing_document
                    else None
                )

                result = ingest_path_sync(
                    path,
                    original_file_name=path.name,
                    mime_type="text/html",
                    source_title=title,
                    source_url=url,
                    source_type="crawler",
                    access_level="public",
                    category="HKPL Website",
                    language="en",
                    effective_date=None,
                    source_kind="crawler",
                    replace_document_id=(
                        str(existing_document["document_id"])
                        if existing_document
                        else None
                    ),
                )

                if (
                    path != previous_path
                    and result.get("status") == "duplicate"
                ):
                    path.unlink(missing_ok=True)

                save_hash(url, content_hash)
                stats["indexed"] += 1
                logger.info("Indexed %s result=%s", url, result)
            else:
                stats["unchanged"] += 1

            if depth < MAX_DEPTH:
                new_links = discover_links(url, raw_html)
                stats["discovered"] += len(new_links)

                for link in new_links:
                    if link not in visited:
                        queue.append((link, depth + 1))

        except Exception as exc:
            stats["failed"] += 1
            logger.exception("Failed to crawl %s: %s", url, exc)

    return stats


if __name__ == "__main__":
    result = crawl()
    print(result)
    sys.exit(0 if result["failed"] == 0 else 1)
