"""Declare supported source extensions without loading parser dependencies."""


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".xlsm",
    ".csv",
    ".md",
    ".txt",
    ".html",
    ".htm",
    ".xml",
    ".json",
    ".jsonl",
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
}

LEGACY_EXTENSIONS = {".doc", ".xls", ".ppt"}
DETERMINISTIC_EXTENSIONS = {".csv", ".xlsx", ".xlsm", ".json", ".jsonl", ".xml"}
DOCLING_EXTENSIONS = SUPPORTED_EXTENSIONS - DETERMINISTIC_EXTENSIONS

