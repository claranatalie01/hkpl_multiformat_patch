"""Fetch and clean one administrator-selected webpage for ingestion.

Unlike the bulk crawler, this helper does no link discovery or change-state
management. It validates one HTTP(S) URL, removes repeated site chrome, saves
the selected main HTML in uploads, and returns it to the admin API for normal
shared ingestion.
"""

import hashlib
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from .html_utils import normalize_html_text

def validate_url(url: str):
    parsed = urlparse(url)

    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only HTTP/HTTPS URLs are supported.")

def fetch_webpage_as_html(url: str) -> tuple[str, str]:
    validate_url(url)

    response = requests.get(
        url,
        timeout=30,
        headers={"User-Agent": "HKPL-RAG-KnowledgeBaseIndexer/1.0"},
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup.select(
        "script, style, nav, footer, header, noscript, svg, "
        ".breadcrumb, .breadcrumbs, .side_menu, .sidebar, "
        ".share, .social, .pagination, .search, .menu"
    ):
        tag.decompose()

    title = (
        normalize_html_text(soup.title.get_text(" ", strip=True))
        if soup.title
        else url
    )

    main = (
        soup.select_one(".event_detail")
        or soup.select_one(".event-details")
        or soup.select_one(".content_detail")
        or soup.select_one(".main_content")
        or soup.select_one("#content")
        or soup.find("main")
        or soup.find("article")
        or soup.body
        or soup
    )

    return title, str(main)


def save_webpage_to_uploads(*, url: str, upload_dir: Path) -> tuple[Path, str]:
    title, html = fetch_webpage_as_html(url)

    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    path = upload_dir / f"webpage_{digest}.html"
    path.write_text(html, encoding="utf-8")

    return path, title
