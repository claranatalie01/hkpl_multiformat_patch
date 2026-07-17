#!/usr/bin/env python3

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import requests
from llama_index.core import Document

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.corpus import (
    DISTRACTOR_CORPUS_ROLE,
    VECTOR_TABLE_NAME,
    replace_dataset_vectors,
)
from src.ingestion.chunking import chunk_documents


DATASET_NAME = "webz_news"
REPOSITORY = "Webhose/free-news-datasets"
REPOSITORY_URL = f"https://github.com/{REPOSITORY}"
TERMS_URL = f"{REPOSITORY_URL}/blob/master/tou.MD"
ARCHIVE_API_URL = (
    f"https://api.github.com/repos/{REPOSITORY}/contents/News_Datasets?ref=master"
)
CACHE_DIR = Path("/app/data/webz_news")
ARCHIVE_TIMESTAMP = re.compile(r"_(\d{14})\.zip$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ingest Webz.io free news articles as distractor noise in the "
            "shared RAG vector table."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser(
        "list",
        help="List available news ZIP archives.",
    )
    list_parser.add_argument("--limit", type=int, default=20)

    prepare = subparsers.add_parser(
        "prepare",
        help="Download one archive and replace the Webz news distractor vectors.",
    )
    prepare.add_argument(
        "--archive",
        dest="archives",
        action="append",
        default=[],
        help=(
            "Exact ZIP filename, local ZIP path, or 'latest'. Repeat to combine "
            "archives; defaults to 'latest'."
        ),
    )
    prepare.add_argument("--limit", type=int, default=1000)
    prepare.add_argument(
        "--language",
        default="",
        help="Optional exact language filter, for example 'english'.",
    )
    prepare.add_argument(
        "--force-download",
        action="store_true",
        help="Replace an already cached archive.",
    )
    prepare.add_argument(
        "--accept-terms",
        action="store_true",
        help=f"Confirm that you reviewed and accept {TERMS_URL}",
    )
    return parser.parse_args()


def github_headers() -> dict[str, str]:
    return {
        "Accept": "application/vnd.github+json",
        "User-Agent": "hkpl-rag-news-ingester",
    }


def request_json(url: str) -> list[dict]:
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            response = requests.get(url, headers=github_headers(), timeout=60)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list):
                raise ValueError("GitHub archive response was not a list.")
            return payload
        except (requests.RequestException, ValueError) as error:
            last_error = error
            if attempt < 3:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Could not list Webz news archives: {last_error}")


def available_archives() -> list[dict]:
    rows = request_json(ARCHIVE_API_URL)
    archives = [
        row
        for row in rows
        if str(row.get("name") or "").lower().endswith(".zip")
        and row.get("download_url")
    ]

    def sort_key(row: dict) -> tuple[str, str]:
        name = str(row["name"])
        match = ARCHIVE_TIMESTAMP.search(name)
        return (match.group(1) if match else "", name)

    return sorted(archives, key=sort_key, reverse=True)


def resolve_archive(value: str) -> tuple[str, str | None]:
    local_path = Path(value).expanduser()
    if local_path.is_file():
        return local_path.name, None

    archives = available_archives()
    if value == "latest":
        if not archives:
            raise RuntimeError("No ZIP archives were found in the repository.")
        selected = archives[0]
    else:
        selected = next(
            (row for row in archives if str(row["name"]) == value),
            None,
        )
        if selected is None:
            raise ValueError(
                f"Archive {value!r} was not found. Run the 'list' command first."
            )
    return str(selected["name"]), str(selected["download_url"])


def download_archive(name: str, url: str, force: bool) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    destination = CACHE_DIR / name
    if destination.is_file() and not force:
        return destination

    partial = destination.with_suffix(destination.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            with requests.get(
                url,
                headers=github_headers(),
                stream=True,
                timeout=(30, 180),
            ) as response:
                response.raise_for_status()
                with partial.open("wb") as output:
                    for block in response.iter_content(chunk_size=1024 * 1024):
                        if block:
                            output.write(block)
            with ZipFile(partial) as archive:
                bad_member = archive.testzip()
                if bad_member:
                    raise BadZipFile(f"Corrupt member: {bad_member}")
            partial.replace(destination)
            return destination
        except (requests.RequestException, OSError, BadZipFile) as error:
            last_error = error
            partial.unlink(missing_ok=True)
            if attempt < 3:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Could not download {name}: {last_error}")


def natural_member_key(name: str) -> tuple:
    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", name)
    )


def clean_article_text(title: str, value: str) -> str:
    lines = [" ".join(line.split()) for line in value.splitlines()]
    cleaned: list[str] = []
    for line in lines:
        if not line:
            continue
        if not cleaned and line.casefold() == title.casefold():
            continue
        if cleaned and line == cleaned[-1]:
            continue
        cleaned.append(line)
    return "\n\n".join(cleaned)


def article_document(record: dict, archive_name: str) -> Document | None:
    title = " ".join(str(record.get("title") or "Untitled news article").split())
    body = clean_article_text(title, str(record.get("text") or ""))
    if len(body) < 50:
        return None

    source_url = str(record.get("url") or "").strip()
    identifier = str(record.get("uuid") or "").strip() or hashlib.sha256(
        f"{source_url}\n{title}\n{body}".encode("utf-8")
    ).hexdigest()[:40]
    document_id = f"webz_news:{identifier}"
    thread = record.get("thread") if isinstance(record.get("thread"), dict) else {}
    metadata = {
        "dataset": DATASET_NAME,
        "corpus": DATASET_NAME,
        "corpus_role": DISTRACTOR_CORPUS_ROLE,
        "document_id": document_id,
        "kb_document_id": document_id,
        "source_title": title,
        "source_url": source_url,
        "source_type": "distractor_benchmark",
        "source_publisher": str(thread.get("site_title") or thread.get("site") or ""),
        "source_repository": REPOSITORY_URL,
        "source_terms_url": TERMS_URL,
        "source_archive": archive_name,
        "document_type": "news",
        "document_version": 1,
        "language": str(record.get("language") or ""),
        "published": str(record.get("published") or ""),
        "author": str(record.get("author") or ""),
        "sentiment": str(record.get("sentiment") or ""),
        "categories": record.get("categories") or [],
        "country": str(thread.get("country") or ""),
        "section_index": 0,
    }
    return Document(
        text=body,
        metadata=metadata,
        excluded_embed_metadata_keys=list(metadata),
        excluded_llm_metadata_keys=list(metadata),
    )


def load_documents(path: Path, limit: int, language: str) -> list[Document]:
    if limit < 1:
        raise ValueError("--limit must be positive.")
    expected_language = language.strip().casefold()
    documents: list[Document] = []
    seen_ids: set[str] = set()
    with ZipFile(path) as archive:
        members = sorted(
            (name for name in archive.namelist() if name.lower().endswith(".json")),
            key=natural_member_key,
        )
        for member in members:
            record = json.loads(archive.read(member))
            if not isinstance(record, dict):
                continue
            record_language = str(record.get("language") or "").casefold()
            if expected_language and record_language != expected_language:
                continue
            document = article_document(record, path.name)
            if document is None:
                continue
            document_id = str(document.metadata["document_id"])
            if document_id in seen_ids:
                continue
            seen_ids.add(document_id)
            documents.append(document)
            if len(documents) >= limit:
                break
    return documents


def list_archives(limit: int) -> None:
    for row in available_archives()[:max(1, limit)]:
        print(f"{row['name']}\t{int(row.get('size') or 0):,} bytes")


def prepare(args: argparse.Namespace) -> None:
    if not args.accept_terms:
        raise RuntimeError(
            "Review the Webz.io dataset terms, then rerun with --accept-terms: "
            f"{TERMS_URL}"
        )

    archive_paths: list[Path] = []
    for archive in args.archives or ["latest"]:
        archive_value = Path(archive).expanduser()
        if archive_value.is_file():
            archive_paths.append(archive_value)
            continue
        archive_name, archive_url = resolve_archive(archive)
        if archive_url is None:
            raise RuntimeError("The selected archive has no download URL.")
        archive_paths.append(download_archive(
            archive_name,
            archive_url,
            args.force_download,
        ))

    documents: list[Document] = []
    seen_ids: set[str] = set()
    for archive_path in archive_paths:
        remaining = args.limit - len(documents)
        if remaining <= 0:
            break
        for document in load_documents(archive_path, remaining, args.language):
            document_id = str(document.metadata["document_id"])
            if document_id in seen_ids:
                continue
            seen_ids.add(document_id)
            documents.append(document)
            if len(documents) >= args.limit:
                break
    if not documents:
        raise RuntimeError("No usable news articles matched the requested selection.")
    nodes = chunk_documents(documents)
    deleted = replace_dataset_vectors(DATASET_NAME, nodes)
    print("Selected archives: " + ", ".join(path.name for path in archive_paths))
    print(f"Removed previous Webz news distractor vectors: {deleted}")
    print(f"News articles ingested: {len(documents)}")
    print(f"News article chunks ingested: {len(nodes)}")
    print(f"Shared vector table: {VECTOR_TABLE_NAME}")
    print("No news questions or expected answers were stored for evaluation.")


def main() -> None:
    args = parse_args()
    if args.command == "list":
        list_archives(args.limit)
    elif args.command == "prepare":
        prepare(args)


if __name__ == "__main__":
    main()
