#!/usr/bin/env python3

import argparse
import shutil
from pathlib import Path
from uuid import uuid4

from src.ingestion.readers import (
    SUPPORTED_EXTENSIONS,
)
from src.ingestion.service import (
    UPLOAD_DIR,
    ingest_path_sync,
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


def ingest_one(path: Path) -> None:
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
        )
        print(f"OK: {result}")
    except Exception as error:
        stored_path.unlink(
            missing_ok=True
        )
        print(
            f"FAILED: {path}: {error}"
        )


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
    args = parser.parse_args()

    target = Path(args.path)

    if target.is_file():
        ingest_one(target)
        return

    if not target.is_dir():
        raise FileNotFoundError(target)

    for file_path in sorted(
        target.rglob("*")
    ):
        if file_path.is_file():
            ingest_one(file_path)


if __name__ == "__main__":
    main()
