"""Environment-backed paths and parser settings shared by ingestion entry points."""

import os
from pathlib import Path


UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/app/uploads"))
OCR_LANGUAGES = os.getenv("OCR_LANGUAGES", "eng+chi_tra+chi_sim")

