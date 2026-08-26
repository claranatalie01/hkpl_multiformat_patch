#!/usr/bin/env python3
"""CLI entry point for ingesting a local file or directory.

Supported files are copied into the durable uploads directory and passed to
``ingest_path_sync``. The shared ingestion service performs registration,
extraction, chunking, embedding, and pgvector insertion.
"""

import argparse
import shutil
from pathlib import Path
from uuid import uuid4

from src.ingestion.readers import (
    SUPPORTED_EXTENSIONS,
)
from src.ingestion.classification import MAX_BATCH_ITEMS
from src.ingestion.service import (
    UPLOAD_DIR,
    ingest_path_sync,
    process_registered_batch,
    register_upload,
)


def copy_into_uploads(path: Path) -> Path:
    UPLOAD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    stored_name = (
        f"{uuid4().hex}_{path.name}"
    )
    destination = (
        UPLOAD_DIR
        / stored_name
    )
    shutil.copy2(
        path,
        destination,
    )
    return destination


def ingest_one(path: Path, document_type: str) -> None:
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        print(
            f"SKIPPED unsupported file: {path}"
        )
        return

    stored_path = copy_into_uploads(path)

    try:
        result = ingest_path_sync(
            stored_path,
            original_file_name=path.name,
            source_title=path.stem,
            source_type="cli_upload",
            document_type=document_type,
        )
        if result.get("status") == "duplicate":
            stored_path.unlink(missing_ok=True)
        print(f"OK: {result}")
    except Exception as error:
        stored_path.unlink(missing_ok=True)
        print(f"FAILED: {path}: {error}")


def register_batch_file(path: Path) -> str | None:
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        print(f"SKIPPED unsupported file: {path}")
        return None

    stored_path = copy_into_uploads(path)
    try:
        registration = register_upload(
            stored_path=stored_path,
            original_file_name=path.name,
            source_title=path.stem,
            source_type="cli_batch",
            source_kind="batch",
            document_type="auto",
        )
    except Exception:
        stored_path.unlink(missing_ok=True)
        raise

    if registration["duplicate"]:
        stored_path.unlink(missing_ok=True)
        print(f"SKIPPED duplicate: {path}")
        return None
    return str(registration["document"]["document_id"])


def ingest_directory(path: Path) -> None:
    document_ids: list[str] = []
    for file_path in sorted(path.rglob("*")):
        if not file_path.is_file():
            continue
        try:
            document_id = register_batch_file(file_path)
            if document_id:
                document_ids.append(document_id)
        except Exception as error:
            print(f"FAILED to register: {file_path}: {error}")

    for start in range(0, len(document_ids), MAX_BATCH_ITEMS):
        batch = document_ids[start:start + MAX_BATCH_ITEMS]
        try:
            for result in process_registered_batch(batch):
                print(f"OK: {result}")
        except Exception as error:
            print(f"FAILED batch {batch}: {error}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Ingest one document or a directory "
            "into the HKPL knowledge base."
        )
    )
    parser.add_argument(
        "path",
        help="Input file or directory",
    )
    parser.add_argument(
        "--document-type",
        choices=("auto", "faq", "record", "prose"),
        default="auto",
        help=(
            "Librarian label for a single file. 'auto' uses 9B content classification. "
            "Directories are always classified in efficient LLM batches."
        ),
    )
    args = parser.parse_args()

    target = Path(args.path)

    if target.is_file():
        ingest_one(target, args.document_type)
        return

    if not target.is_dir():
        raise FileNotFoundError(target)

    if args.document_type != "auto":
        parser.error("--document-type applies only to a single file")
    ingest_directory(target)


if __name__ == "__main__":
    main()
