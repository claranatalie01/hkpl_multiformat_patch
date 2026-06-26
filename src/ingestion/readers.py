import csv
import hashlib
import io
import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable

import fitz
import openpyxl
import pytesseract
from bs4 import BeautifulSoup
from docx import Document as WordDocument
from lxml import etree
from PIL import Image
from pptx import Presentation

from llama_index.core import Document


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

LEGACY_EXTENSIONS = {
    ".doc",
    ".xls",
    ".ppt",
}


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\u00a0", " ")
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def detect_document_type(text: str) -> str:
    lower = text.lower()

    if "question:" in lower and "answer:" in lower:
        return "faq"

    if "district" in lower and "public library" in lower and "tel:" in lower:
        return "directory"

    if any(word in lower for word in ["announcement", "notice", "temporary closure", "suspension"]):
        return "announcement"

    if any(word in lower for word in ["policy", "rules", "regulations", "guidelines"]):
        return "policy"

    return "general"

def file_content_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _base_metadata(
    *,
    document_id: str,
    original_file_name: str,
    stored_file_name: str,
    source_title: str,
    source_url: str,
    source_type: str,
    access_level: str,
    document_version: int,
    content_hash: str,
    file_type: str,
    category: str | None = None,
    language: str | None = None,
    effective_date: str | None = None,
    source_kind: str = "upload",
) -> dict:
    return {
        "document_id": document_id,
        "original_file_name": original_file_name,
        "file_name": original_file_name,
        "stored_file_name": stored_file_name,
        "file_type": file_type,
        "source_title": source_title or Path(original_file_name).stem,
        "source_url": source_url or "",
        "url": source_url or "",
        "source_type": source_type,
        "access_level": access_level,
        "document_version": document_version,
        "content_hash": content_hash,
        "category": category or "",
        "language": language or "",
        "effective_date": effective_date or "",
        "source_kind": source_kind,
    }


def _make_document(
    text: str,
    *,
    base_metadata: dict,
    section_index: int,
    chunk_strategy: str = "prose",
    extra_metadata: dict | None = None,
) -> Document | None:
    clean = normalize_text(text)
    if not clean:
        return None

    metadata = {
        **base_metadata,
        "section_index": section_index,
        "chunk_strategy": chunk_strategy,
    }

    if extra_metadata:
        metadata.update(extra_metadata)

    metadata["document_type"] = metadata.get("document_type") or detect_document_type(clean)

    document = Document(
        text=clean,
        metadata=metadata,
    )

    document.id_ = (
        f"{base_metadata['document_id']}:"
        f"v{base_metadata['document_version']}:"
        f"section:{section_index}"
    )

    return document


def _ocr_image(image: Image.Image, languages: str) -> str:
    return normalize_text(
        pytesseract.image_to_string(
            image,
            lang=languages,
            config="--psm 6",
        )
    )

def _extract_pdf_text_with_layout_markers(page) -> str:
    data = page.get_text("dict")
    lines = []

    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue

            text = normalize_text(" ".join(span.get("text", "") for span in spans))
            if not text:
                continue

            max_size = max(span.get("size", 0) for span in spans)
            is_bold = any("bold" in span.get("font", "").lower() for span in spans)

            lines.append(
                {
                    "text": text,
                    "font_size": max_size,
                    "is_bold": is_bold,
                }
            )

    if not lines:
        return ""

    sizes = sorted(line["font_size"] for line in lines)
    median_size = sizes[len(sizes) // 2]

    output_lines = []

    for item in lines:
        text = item["text"]
        word_count = len(text.split())
        has_terminal_punctuation = text.endswith((".", "?", "!", ":"))
        has_many_digits = sum(ch.isdigit() for ch in text) >= 3

        heading_score = 0

        if item["font_size"] > median_size + 1:
            heading_score += 2

        if item["is_bold"]:
            heading_score += 1

        if word_count <= 6:
            heading_score += 1

        if not has_terminal_punctuation:
            heading_score += 1

        if not has_many_digits:
            heading_score += 1

        # Layout/typography-based heading detection.
        # Not based on district names or specific HKPL words.
        if heading_score >= 4:
            output_lines.append(f"## {text}")
        else:
            output_lines.append(text)

    return normalize_text("\n".join(output_lines))

def _load_pdf(
    path: Path,
    base_metadata: dict,
    ocr_languages: str,
) -> list[Document]:
    documents: list[Document] = []
    pdf = fitz.open(path)

    try:
        for page_index, page in enumerate(pdf):
            page_number = page_index + 1

            text = _extract_pdf_text_with_layout_markers(page)
            extraction_method = "native_layout_text"

            if len(text) < 40:
                pixmap = page.get_pixmap(
                    matrix=fitz.Matrix(2, 2),
                    alpha=False,
                )
                image = Image.open(
                    io.BytesIO(pixmap.tobytes("png"))
                )
                ocr_text = _ocr_image(image, ocr_languages)

                if len(ocr_text) > len(text):
                    text = ocr_text
                    extraction_method = "ocr"

            document = _make_document(
                text,
                base_metadata=base_metadata,
                section_index=page_index,
                chunk_strategy="prose",
                extra_metadata={
                    "page_number": page_number,
                    "extraction_method": extraction_method,
                },
            )

            if document:
                documents.append(document)
    finally:
        pdf.close()

    return documents


def _load_docx(
    path: Path,
    base_metadata: dict,
) -> list[Document]:
    word_document = WordDocument(path)
    documents: list[Document] = []

    current_heading = ""
    current_parts: list[str] = []
    section_index = 0

    def flush_section() -> None:
        nonlocal section_index, current_parts
        if not current_parts:
            return

        content = "\n".join(current_parts)
        document = _make_document(
            content,
            base_metadata=base_metadata,
            section_index=section_index,
            chunk_strategy="prose",
            extra_metadata={
                "section_heading": current_heading,
                "content_kind": "paragraphs",
            },
        )
        if document:
            documents.append(document)
            section_index += 1
        current_parts = []

    for paragraph in word_document.paragraphs:
        text = normalize_text(paragraph.text)
        if not text:
            continue

        style_name = (
            paragraph.style.name.lower()
            if paragraph.style and paragraph.style.name
            else ""
        )

        if style_name.startswith("heading"):
            flush_section()
            current_heading = text
            current_parts = [text]
        else:
            current_parts.append(text)

    flush_section()

    for table_index, table in enumerate(word_document.tables):
        rows = []
        for row in table.rows:
            cells = [
                normalize_text(cell.text)
                for cell in row.cells
            ]
            rows.append(" | ".join(cells))

        document = _make_document(
            "\n".join(rows),
            base_metadata=base_metadata,
            section_index=section_index,
            chunk_strategy="atomic",
            extra_metadata={
                "table_index": table_index,
                "content_kind": "table",
            },
        )
        if document:
            documents.append(document)
            section_index += 1

    return documents


def _load_pptx(
    path: Path,
    base_metadata: dict,
) -> list[Document]:
    presentation = Presentation(path)
    documents: list[Document] = []

    for slide_index, slide in enumerate(presentation.slides):
        parts = []
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text = normalize_text(shape.text)
                if text:
                    parts.append(text)

        document = _make_document(
            "\n".join(parts),
            base_metadata=base_metadata,
            section_index=slide_index,
            chunk_strategy="prose",
            extra_metadata={
                "slide_number": slide_index + 1,
            },
        )
        if document:
            documents.append(document)

    return documents


def _load_csv(
    path: Path,
    base_metadata: dict,
) -> list[Document]:
    documents: list[Document] = []

    with path.open(
        newline="",
        encoding="utf-8-sig",
        errors="replace",
    ) as file:
        reader = csv.DictReader(file)

        for row_index, row in enumerate(reader):
            fields = [
                f"{normalize_text(str(column))}: "
                f"{normalize_text(str(value))}"
                for column, value in row.items()
                if column is not None
                and value is not None
                and normalize_text(str(value))
            ]

            document = _make_document(
                "\n".join(fields),
                base_metadata=base_metadata,
                section_index=row_index,
                chunk_strategy="atomic",
                extra_metadata={
                    "row_number": row_index + 2,
                    "content_kind": "table_row",
                },
            )
            if document:
                documents.append(document)

    return documents


def _load_excel(
    path: Path,
    base_metadata: dict,
) -> list[Document]:
    workbook = openpyxl.load_workbook(
        path,
        data_only=True,
        read_only=True,
    )
    documents: list[Document] = []
    section_index = 0

    for worksheet in workbook.worksheets:
        iterator = worksheet.iter_rows(values_only=True)

        try:
            first_row = next(iterator)
        except StopIteration:
            continue

        headers = [
            normalize_text(str(value)) if value is not None else ""
            for value in first_row
        ]

        for row_number, row in enumerate(iterator, start=2):
            values = [
                normalize_text(str(value)) if value is not None else ""
                for value in row
            ]

            fields = [
                f"{header}: {value}"
                for header, value in zip(headers, values)
                if header and value
            ]

            document = _make_document(
                "\n".join(fields),
                base_metadata=base_metadata,
                section_index=section_index,
                chunk_strategy="atomic",
                extra_metadata={
                    "sheet_name": worksheet.title,
                    "row_number": row_number,
                    "content_kind": "table_row",
                },
            )
            if document:
                documents.append(document)
                section_index += 1

    workbook.close()
    return documents


def _split_markdown_sections(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    heading = ""
    parts: list[str] = []

    for line in text.splitlines():
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if match:
            if parts:
                sections.append((heading, "\n".join(parts)))
            heading = normalize_text(match.group(2))
            parts = [line]
        else:
            parts.append(line)

    if parts:
        sections.append((heading, "\n".join(parts)))

    return sections


def _load_text_or_markdown(
    path: Path,
    base_metadata: dict,
) -> list[Document]:
    raw = path.read_text(encoding="utf-8", errors="replace")

    if path.suffix.lower() == ".md":
        sections = _split_markdown_sections(raw)
    else:
        sections = [
            ("", section)
            for section in re.split(r"\n\s*\n", raw)
            if normalize_text(section)
        ]

    documents: list[Document] = []
    for section_index, (heading, text) in enumerate(sections):
        document = _make_document(
            text,
            base_metadata=base_metadata,
            section_index=section_index,
            chunk_strategy="prose",
            extra_metadata={
                "section_heading": heading,
            },
        )
        if document:
            documents.append(document)

    return documents


def _load_html(
    path: Path,
    base_metadata: dict,
) -> list[Document]:
    html = path.read_text(
        encoding="utf-8",
        errors="replace",
    )
    soup = BeautifulSoup(html, "html.parser")

    for element in soup(
        [
            "script",
            "style",
            "nav",
            "footer",
            "noscript",
            "svg",
        ]
    ):
        element.decompose()

    page_title = (
        normalize_text(soup.title.get_text(" ", strip=True))
        if soup.title
        else base_metadata["source_title"]
    )

    documents: list[Document] = []
    current_heading = page_title
    current_parts: list[str] = []
    section_index = 0

    def flush() -> None:
        nonlocal section_index, current_parts
        if not current_parts:
            return

        document = _make_document(
            "\n".join(current_parts),
            base_metadata={
                **base_metadata,
                "source_title": page_title,
            },
            section_index=section_index,
            chunk_strategy="prose",
            extra_metadata={
                "section_heading": current_heading,
            },
        )
        if document:
            documents.append(document)
            section_index += 1
        current_parts = []

    body = soup.body or soup

    for element in body.find_all(
        ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "table"]
    ):
        text = normalize_text(element.get_text(" ", strip=True))
        if not text:
            continue

        if element.name and element.name.startswith("h"):
            flush()
            current_heading = text
            current_parts = [text]
        else:
            current_parts.append(text)

    flush()

    if not documents:
        document = _make_document(
            soup.get_text("\n", strip=True),
            base_metadata={
                **base_metadata,
                "source_title": page_title,
            },
            section_index=0,
            chunk_strategy="prose",
        )
        if document:
            documents.append(document)

    return documents


def _load_xml(
    path: Path,
    base_metadata: dict,
) -> list[Document]:
    parser = etree.XMLParser(
        recover=True,
        resolve_entities=False,
        no_network=True,
        huge_tree=False,
    )
    tree = etree.parse(str(path), parser)
    root = tree.getroot()

    children = list(root)
    targets = children if children else [root]

    documents: list[Document] = []

    for section_index, element in enumerate(targets):
        text_parts = [
            normalize_text(text)
            for text in element.itertext()
            if normalize_text(text)
        ]
        tag = etree.QName(element.tag).localname

        document = _make_document(
            "\n".join(text_parts),
            base_metadata=base_metadata,
            section_index=section_index,
            chunk_strategy="atomic",
            extra_metadata={
                "xml_tag": tag,
            },
        )
        if document:
            documents.append(document)

    return documents


def _json_record_to_text(record: object) -> str:
    if isinstance(record, dict):
        return "\n".join(
            f"{normalize_text(str(key))}: "
            f"{normalize_text(json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value))}"
            for key, value in record.items()
        )

    if isinstance(record, list):
        return "\n".join(
            normalize_text(str(item))
            for item in record
        )

    return normalize_text(str(record))


def _load_json(
    path: Path,
    base_metadata: dict,
) -> list[Document]:
    documents: list[Document] = []

    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open(
            encoding="utf-8",
            errors="replace",
        ) as file:
            for line in file:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    else:
        data = json.loads(
            path.read_text(
                encoding="utf-8",
                errors="replace",
            )
        )
        records = data if isinstance(data, list) else [data]

    for record_index, record in enumerate(records):
        document = _make_document(
            _json_record_to_text(record),
            base_metadata=base_metadata,
            section_index=record_index,
            chunk_strategy="atomic",
            extra_metadata={
                "record_index": record_index,
            },
        )
        if document:
            documents.append(document)

    return documents


def _load_image(
    path: Path,
    base_metadata: dict,
    ocr_languages: str,
) -> list[Document]:
    with Image.open(path) as image:
        text = _ocr_image(image.convert("RGB"), ocr_languages)

    document = _make_document(
        text,
        base_metadata=base_metadata,
        section_index=0,
        chunk_strategy="prose",
        extra_metadata={
            "extraction_method": "ocr",
            "ocr_languages": ocr_languages,
        },
    )

    return [document] if document else []


def load_file(
    path: Path,
    *,
    document_id: str,
    original_file_name: str,
    source_title: str = "",
    source_url: str = "",
    source_type: str = "admin_upload",
    access_level: str = "public",
    document_version: int = 1,
    content_hash: str | None = None,
    ocr_languages: str = "eng+chi_tra",
    category: str | None = None,
    language: str | None = None,
    effective_date: str | None = None,
    source_kind: str = "upload",
) -> list[Document]:
    
    extension = path.suffix.lower()

    if extension in LEGACY_EXTENSIONS:
        raise ValueError(
            f"Legacy format {extension} is not supported directly. "
            "Convert it to .docx, .xlsx, or .pptx first."
        )

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension: {extension}"
        )

    actual_hash = content_hash or file_content_hash(path)

    base_metadata = _base_metadata(
        document_id=document_id,
        original_file_name=original_file_name,
        stored_file_name=path.name,
        source_title=source_title,
        source_url=source_url,
        source_type=source_type,
        access_level=access_level,
        document_version=document_version,
        content_hash=actual_hash,
        file_type=extension.lstrip("."),
        category=category or "",
        language=language or "",
        effective_date=effective_date or "",
        source_kind=source_kind,
    )

    if extension == ".pdf":
        return _load_pdf(path, base_metadata, ocr_languages)
    if extension == ".docx":
        return _load_docx(path, base_metadata)
    if extension == ".pptx":
        return _load_pptx(path, base_metadata)
    if extension == ".csv":
        return _load_csv(path, base_metadata)
    if extension in {".xlsx", ".xlsm"}:
        return _load_excel(path, base_metadata)
    if extension in {".md", ".txt"}:
        return _load_text_or_markdown(path, base_metadata)
    if extension in {".html", ".htm"}:
        return _load_html(path, base_metadata)
    if extension == ".xml":
        return _load_xml(path, base_metadata)
    if extension in {".json", ".jsonl"}:
        return _load_json(path, base_metadata)
    if extension in {
        ".jpg",
        ".jpeg",
        ".png",
        ".tif",
        ".tiff",
    }:
        return _load_image(
            path,
            base_metadata,
            ocr_languages,
        )

    raise ValueError(f"No reader configured for {extension}")
