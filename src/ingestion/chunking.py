import hashlib
import os
import re
from typing import List

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode
from .document_types import chunk_strategy_for, detect_document_type


PROSE_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
PROSE_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))
ATOMIC_MAX_TOKENS = int(os.getenv("ATOMIC_MAX_TOKENS", "2048"))
RECORD_MAX_TOKENS = int(os.getenv("RECORD_MAX_TOKENS", "8192"))

prose_splitter = SentenceSplitter(
    chunk_size=PROSE_CHUNK_SIZE,
    chunk_overlap=PROSE_CHUNK_OVERLAP,
)

atomic_splitter = SentenceSplitter(
    chunk_size=ATOMIC_MAX_TOKENS,
    chunk_overlap=0,
)

record_splitter = SentenceSplitter(
    chunk_size=RECORD_MAX_TOKENS,
    chunk_overlap=0,
)


def get_text(document: Document) -> str:
    return getattr(document, "text", None) or document.get_content()


def choose_chunking_strategy(document: Document) -> str:
    text = get_text(document)
    metadata = document.metadata or {}
    doc_type = detect_document_type(text, metadata)
    return chunk_strategy_for(doc_type)


def split_faq_entries(document: Document) -> List[Document]:
    text = get_text(document)
    metadata = document.metadata or {}
    matches = list(re.finditer(r"(?im)^\s*Q\d+\s*[:.)]", text))
    if not matches:
        document.metadata.update({
            "chunk_strategy": "atomic",
            "document_type": "faq",
        })
        return [document]

    heading = text[:matches[0].start()].strip()
    output = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        entry = text[match.start():end].strip()
        if not entry:
            continue
        content = f"{heading}\n\n{entry}" if heading else entry
        output.append(Document(text=content, metadata={
            **metadata,
            "section_index": index,
            "chunk_strategy": "atomic",
            "document_type": "faq",
            "faq_entry_index": index,
        }))
    return output or [document]


def split_announcement_entries(document: Document) -> List[Document]:
    text = get_text(document)
    metadata = document.metadata or {}
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    dated_entry = re.compile(
        r"^\(?\s*\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\s*\)?(?:\s+|$)",
        re.IGNORECASE,
    )
    entry_starts = [index for index, line in enumerate(lines) if dated_entry.match(line)]

    # A detailed announcement page is already one coherent item. Only listing
    # pages with multiple dated entries need to be separated.
    if len(entry_starts) < 2:
        document.metadata.update({
            "chunk_strategy": "atomic",
            "document_type": "announcement",
        })
        return [document]

    heading = "\n".join(lines[:entry_starts[0]]).strip()
    output = []
    for entry_index, start in enumerate(entry_starts):
        end = entry_starts[entry_index + 1] if entry_index + 1 < len(entry_starts) else len(lines)
        entry = "\n".join(lines[start:end]).strip()
        if not entry:
            continue
        content = f"{heading}\n\n{entry}" if heading else entry
        output.append(Document(text=content, metadata={
            **metadata,
            "section_index": entry_index,
            "chunk_strategy": "atomic",
            "document_type": "announcement",
            "announcement_entry_index": entry_index,
        }))
    return output or [document]


def split_marked_sections(document: Document) -> List[Document]:
    text = get_text(document)
    metadata = document.metadata or {}

    sections = []
    current_heading = ""
    current_lines = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("## "):
            if current_lines:
                sections.append((current_heading, "\n".join(current_lines)))

            current_heading = line.replace("## ", "", 1).strip()
            current_lines = [current_heading]
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_heading, "\n".join(current_lines)))

    output = []

    for section_index, (heading, section_text) in enumerate(sections):
        normalized_heading = " ".join((heading or "").split())
        normalized_text = " ".join(section_text.split())

        if normalized_heading and normalized_text == normalized_heading:
            continue

        if len(normalized_text.split()) < 5:
            continue

        output.append(
            Document(
                text=section_text,
                metadata={
                    **metadata,
                    "section_heading": heading,
                    "section_index": section_index,
                    "chunk_strategy": "atomic",
                    "document_type": "structured",
                },
            )
        )

    return output or [document]


def prepare_documents_for_chunking(documents: List[Document]) -> List[Document]:
    prepared = []

    for document in documents:
        text = get_text(document)
        metadata = document.metadata or {}
        strategy = choose_chunking_strategy(document)
        doc_type = detect_document_type(text, metadata)

        document.metadata.update(
            {
                **metadata,
                "dataset": metadata.get("dataset") or "hkpl",
                "corpus": metadata.get("corpus") or "hkpl",
                "corpus_role": metadata.get("corpus_role") or "primary",
                "chunk_strategy": strategy,
                "document_type": doc_type,
            }
        )

        if strategy == "faq_entries":
            prepared.extend(split_faq_entries(document))
        elif strategy == "announcement_entries":
            prepared.extend(split_announcement_entries(document))
        elif strategy == "directory_sections":
            prepared.extend(split_directory_entries(document))
        elif strategy == "marked_sections":
            prepared.extend(split_marked_sections(document))
        else:
            prepared.append(document)

    return prepared


def chunk_documents(documents: List[Document]) -> List[BaseNode]:
    prepared_documents = prepare_documents_for_chunking(documents)
    nodes: list[BaseNode] = []
    seen_document_content: set[tuple[str, str]] = set()

    for document in prepared_documents:
        strategy = document.metadata.get("chunk_strategy", "prose")
        document_type = document.metadata.get("document_type", "prose")
        if strategy == "atomic" and document_type == "record_based":
            splitter = record_splitter
        elif strategy == "atomic":
            splitter = atomic_splitter
        else:
            splitter = prose_splitter

        document_nodes = splitter.get_nodes_from_documents([document])

        kb_document_id = str(
            document.metadata.get("kb_document_id")
            or document.metadata.get("document_id", "")
        )

        for local_index, node in enumerate(document_nodes):
            content = node.get_content()

            if document_type == "news":
                title = str(node.metadata.get("source_title") or "").strip()
                title_line = f"Title: {title}" if title else ""
                if title_line and not content.startswith(title_line):
                    content = f"{title_line}\n\n{content}"
                    node.text = content

            if len(content.strip()) < 50:
                continue

            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            deduplication_key = (kb_document_id, content_hash)
            if deduplication_key in seen_document_content:
                continue
            seen_document_content.add(deduplication_key)

            digest = content_hash[:16]

            version = node.metadata["document_version"]
            section_index = node.metadata.get("section_index", 0)

            node.id_ = (
                f"{kb_document_id}:v{version}:"
                f"s{section_index}:c{local_index}:{digest}"
            )

            node.metadata.update(
                {
                    "kb_document_id": kb_document_id,
                    "chunk_id": node.id_,
                    "chunk_index": local_index,
                    "chunk_strategy": strategy,
                    "chunk_size": (
                        RECORD_MAX_TOKENS
                        if strategy == "atomic" and document_type == "record_based"
                        else ATOMIC_MAX_TOKENS
                        if strategy == "atomic"
                        else PROSE_CHUNK_SIZE
                    ),
                    "chunk_overlap": 0 if strategy == "atomic" else PROSE_CHUNK_OVERLAP,
                }
            )

            nodes.append(node)

    return nodes

def clean_marker(line: str) -> str:
    return line.strip().replace("## ", "", 1).strip()

def looks_like_library_entry(line: str) -> bool:
    line = clean_marker(line)
    return (
        len(line.split()) <= 8
        and "library" in line.lower()
        and not line.lower().startswith("hong kong public libraries")
    )


def looks_like_area_heading(line: str) -> bool:
    raw = line.strip()
    line = clean_marker(line)
    lower = line.lower()

    if not line:
        return False

    if "## " not in raw:
        return False

    if any(x in lower for x in [
        "library", "tel", "road", "street", "building",
        "floor", "estate", "services", "enquiries", "website",
    ]):
        return False

    if len(line.split()) <= 6 and not any(ch.isdigit() for ch in line):
        return True

    return False

def split_directory_entries(document: Document) -> List[Document]:
    text = get_text(document)
    metadata = document.metadata or {}

    output: list[Document] = []

    current_area = ""
    current_library = ""
    current_lines: list[str] = []
    section_index = 0

    def flush_entry():
        nonlocal section_index, current_library, current_lines

        if not current_library or not current_lines:
            return

        entry_text = "\n".join(current_lines)

        output.append(
            Document(
                text=entry_text,
                metadata={
                    **metadata,
                    "section_heading": current_area,
                    "directory_area": current_area,
                    "library_name": current_library,
                    "section_index": section_index,
                    "chunk_strategy": "atomic",
                    "document_type": "directory",
                },
            )
        )

        section_index += 1
        current_library = ""
        current_lines = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if looks_like_area_heading(line):
            flush_entry()
            current_area = clean_marker(line)
            continue

        if looks_like_library_entry(line):
            flush_entry()
            current_library = clean_marker(line)
            current_lines = [
                f"Area: {current_area}",
                f"Library: {current_library}",
            ]
            continue

        if current_library:
            current_lines.append(line)

    flush_entry()

    return output or [document]
