#!/usr/bin/env python3

import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import hashlib
import logging
import re
from collections import deque
from urllib.parse import urldefrag, urljoin, urlparse, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

from src.ingestion.registry import find_active_web_document_by_source_url
from src.ingestion.service import UPLOAD_DIR, ingest_path_sync

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hkpl_crawler")

DEFAULT_SEED_URLS = [
    "https://www.hkpl.gov.hk/en/index.html",
    "https://www.hkpl.gov.hk/en/extension-activities/",
    "https://www.hkpl.gov.hk/en/about-us/services/",
    "https://www.hkpl.gov.hk/en/e-resources/",
    "https://www.hkpl.gov.hk/en/collections/",
    "https://www.hkpl.gov.hk/en/locations/libraries.html",
    "https://www.hkpl.gov.hk/en/library-notices/library-notices-list.html",
    "https://www.hkpl.gov.hk/en/ask-a-librarian/faq.html",
]

USER_AGENT = "HKPL-RAG-Crawler/1.0"
DEFAULT_MAX_PAGES = int(os.getenv("HKPL_CRAWLER_MAX_PAGES", "300"))
DEFAULT_MAX_DEPTH = int(os.getenv("HKPL_CRAWLER_MAX_DEPTH", "3"))
DEFAULT_DELAY_SECONDS = float(os.getenv("HKPL_CRAWLER_DELAY_SECONDS", "0.5"))
TIMEOUT = int(os.getenv("HKPL_CRAWLER_TIMEOUT_SECONDS", "30"))

CRAWL_STATE_DIR = Path("/app/storage/crawler_state")
CRAWL_STATE_DIR.mkdir(parents=True, exist_ok=True)


def env_flag(name: str, default: bool) -> bool:
    fallback = "true" if default else "false"
    return os.getenv(name, fallback).strip().lower() in {"1", "true", "yes", "on"}


def normalize_url(url: str, *, include_query_urls: bool = False) -> str:
    url, _fragment = urldefrag(url)
    parsed = urlsplit(url.strip())
    query = parsed.query if include_query_urls else ""
    return urlunsplit(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path or "/",
            query,
            "",
        )
    )


def is_allowed_url(url: str, *, include_query_urls: bool = False) -> bool:
    parsed = urlparse(url)
    host = parsed.hostname.lower() if parsed.hostname else ""
    path = parsed.path.lower()

    if parsed.scheme not in {"http", "https"}:
        return False

    # Keep crawler focused on the main HKPL website.
    # This is domain-level control, not page-specific hardcoding.
    if host != "www.hkpl.gov.hk":
        return False

    if parsed.query and not include_query_urls:
        return False

    is_pdf = path.endswith(".pdf")
    if not is_pdf and not path.startswith("/en/"):
        return False

    blocked_paths = (
        "/hkap/",
        "/patron/",
        "/search",
        "/form.html",
    )
    if any(part in path for part in blocked_paths):
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


def save_for_ingestion(url: str, content: str | bytes, extension: str) -> Path:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    path = UPLOAD_DIR / f"crawler_{digest}{extension}"
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")
    return path


def fetch(session: requests.Session, url: str) -> requests.Response:
    response = session.get(
        url,
        timeout=TIMEOUT,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    return response


def discover_links(
    base_url: str,
    html: str,
    *,
    include_query_urls: bool,
) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = []

    for tag in soup.find_all("a", href=True):
        absolute = normalize_url(
            urljoin(base_url, tag["href"]),
            include_query_urls=include_query_urls,
        )
        if is_allowed_url(absolute, include_query_urls=include_query_urls):
            links.append(absolute)

    return sorted(set(links))


def robots_policy() -> RobotFileParser:
    policy = RobotFileParser()
    policy.set_url("https://www.hkpl.gov.hk/robots.txt")
    try:
        policy.read()
    except Exception:
        logger.warning("Could not read robots.txt; refusing to crawl.", exc_info=True)
        policy.parse(["User-agent: *", "Disallow: /"])
    return policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crawl public English HKPL pages and ingest them into PGVector."
    )
    parser.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES)
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    parser.add_argument("--delay-seconds", type=float, default=DEFAULT_DELAY_SECONDS)
    parser.add_argument(
        "--seed-url",
        action="append",
        dest="seed_urls",
        help="Repeat to replace the default seed URL list.",
    )
    parser.add_argument(
        "--include-query-urls",
        action="store_true",
        default=env_flag("HKPL_CRAWLER_INCLUDE_QUERY_URLS", False),
        help="Include query-string URLs. Disabled by default to avoid search-result explosions.",
    )
    parser.add_argument(
        "--exclude-pdfs",
        action="store_true",
        default=not env_flag("HKPL_CRAWLER_INCLUDE_PDFS", True),
        help="Do not download linked HKPL PDF documents.",
    )
    args = parser.parse_args()
    if args.max_pages < 1 or args.max_depth < 0 or args.delay_seconds < 0:
        parser.error("max-pages must be positive; max-depth and delay must be non-negative")
    return args


def crawl(
    *,
    seed_urls: list[str],
    max_pages: int,
    max_depth: int,
    delay_seconds: float,
    include_query_urls: bool,
    include_pdfs: bool,
) -> dict:
    visited = set()
    queue = deque(
        (
            normalize_url(url, include_query_urls=include_query_urls),
            0,
        )
        for url in seed_urls
    )
    policy = robots_policy()
    session = requests.Session()

    stats = {
        "visited": 0,
        "indexed": 0,
        "unchanged": 0,
        "failed": 0,
        "discovered": 0,
        "html_indexed": 0,
        "pdf_indexed": 0,
        "skipped_robots": 0,
        "skipped_unsupported": 0,
    }

    while queue and stats["visited"] < max_pages:
        url, depth = queue.popleft()
        url = normalize_url(url, include_query_urls=include_query_urls)

        if url in visited:
            continue

        if not is_allowed_url(url, include_query_urls=include_query_urls):
            continue

        if not policy.can_fetch(USER_AGENT, url):
            stats["skipped_robots"] += 1
            continue

        visited.add(url)
        stats["visited"] += 1

        try:
            logger.info("Fetching depth=%s url=%s", depth, url)
            response = fetch(session, url)
            content_type = response.headers.get("content-type", "").lower()
            is_pdf = urlparse(url).path.lower().endswith(".pdf") or (
                "application/pdf" in content_type
            )
            existing_document = find_active_web_document_by_source_url(url)

            if is_pdf:
                if not include_pdfs:
                    stats["skipped_unsupported"] += 1
                    continue
                content_hash = hashlib.sha256(response.content).hexdigest()
                title = Path(urlparse(url).path).name or "HKPL PDF"
                saved_content = response.content
                extension = ".pdf"
                mime_type = "application/pdf"
            elif "html" in content_type or not content_type:
                raw_html = response.text
                title, main_html = extract_main_html(raw_html)
                content_hash = page_hash(main_html)
                saved_content = main_html
                extension = ".html"
                mime_type = "text/html"
            else:
                stats["skipped_unsupported"] += 1
                continue

            if existing_document is None or has_changed(url, content_hash):
                path = save_for_ingestion(url, saved_content, extension)
                previous_path = (
                    UPLOAD_DIR / existing_document["stored_file_name"]
                    if existing_document
                    else None
                )

                result = ingest_path_sync(
                    path,
                    original_file_name=path.name,
                    mime_type=mime_type,
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
                if is_pdf:
                    stats["pdf_indexed"] += 1
                else:
                    stats["html_indexed"] += 1
                logger.info("Indexed %s result=%s", url, result)
            else:
                stats["unchanged"] += 1

            if not is_pdf and depth < max_depth:
                new_links = discover_links(
                    url,
                    raw_html,
                    include_query_urls=include_query_urls,
                )
                stats["discovered"] += len(new_links)

                for link in new_links:
                    if link not in visited:
                        queue.append((link, depth + 1))

        except Exception as exc:
            stats["failed"] += 1
            logger.exception("Failed to crawl %s: %s", url, exc)
        finally:
            if delay_seconds:
                time.sleep(delay_seconds)

    return stats


if __name__ == "__main__":
    arguments = parse_args()
    result = crawl(
        seed_urls=arguments.seed_urls or DEFAULT_SEED_URLS,
        max_pages=arguments.max_pages,
        max_depth=arguments.max_depth,
        delay_seconds=arguments.delay_seconds,
        include_query_urls=arguments.include_query_urls,
        include_pdfs=not arguments.exclude_pdfs,
    )
    print(result)
    successful_pages = result["indexed"] + result["unchanged"]
    sys.exit(0 if successful_pages > 0 else 1)
