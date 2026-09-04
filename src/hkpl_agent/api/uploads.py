"""Validate and persist bounded administrator uploads before ingestion."""

import os
import re
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, UploadFile

from ..ingestion.config import UPLOAD_DIR
from ..ingestion.formats import SUPPORTED_EXTENSIONS


MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))


def safe_filename(filename: str) -> str:
    """Return a basename containing only portable filename characters."""

    basename = Path(filename).name
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", basename)
    return cleaned or "uploaded_file"


def validate_file_signature(extension: str, content: bytes) -> None:
    """Reject empty files and known binary-extension/signature mismatches."""

    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    signatures = {
        ".pdf": [b"%PDF"],
        ".png": [b"\x89PNG\r\n\x1a\n"],
        ".jpg": [b"\xff\xd8\xff"],
        ".jpeg": [b"\xff\xd8\xff"],
        ".tif": [b"II*\x00", b"MM\x00*"],
        ".tiff": [b"II*\x00", b"MM\x00*"],
        ".docx": [b"PK\x03\x04"],
        ".xlsx": [b"PK\x03\x04"],
        ".xlsm": [b"PK\x03\x04"],
        ".pptx": [b"PK\x03\x04"],
    }
    expected = signatures.get(extension)
    if expected and not any(content.startswith(signature) for signature in expected):
        raise HTTPException(
            status_code=400,
            detail=f"The file content does not match the {extension} extension.",
        )


async def save_upload(file: UploadFile) -> tuple[Path, str, str]:
    """Validate an upload and save it under a collision-resistant name."""

    original_name = safe_filename(file.filename or "")
    extension = Path(original_name).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type: {extension}. "
                f"Allowed: {sorted(SUPPORTED_EXTENSIONS)}"
            ),
        )

    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Uploaded file is too large.")

    validate_file_signature(extension, content)
    stored_path = UPLOAD_DIR / f"{uuid4().hex}_{original_name}"
    stored_path.write_bytes(content)
    return stored_path, original_name, file.content_type or ""
