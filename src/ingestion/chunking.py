import hashlib
import os
import re
from typing import List

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode


PROSE_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
PROSE_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))
ATOMIC_MAX_TOKENS = int(os.getenv("ATOMIC_MAX_TOKENS", "2048"))

prose_splitter = SentenceSplitter(
    chunk_size=PROSE_CHUNK_SIZE,
    chunk_overlap=PROSE_CHUNK_OVERLAP,
)

atomic_splitter = SentenceSplitter(
    chunk_size=ATOMIC_MAX_TOKENS,
    chunk_overlap=0,
)


def get_text(document: Document) -> str:
    return getattr(document, "text", None) or document.get_content()


def detect_document_type(text: str, metadata: dict | None = None) -> str:
    metadata = metadata or {}
    lower = text.lower()

    if metadata.get("question") or ("question:" in lower and "answer:" in lower):
        return "faq"

    if metadata.get("file_type") in ["csv", "xlsx", "xlsm", "json", "jsonl", "xml"]:
        return "record_based"

    if "## " in text:
        return "structured"

    if any(x in lower for x in ["announcement", "notice", "temporary closure", "suspension"]):
        return "announcement"

    return "prose"


def choose_chunking_strategy(document: Document) -> str:
    text = get_text(document)
    metadata = document.metadata or {}
    file_type = metadata.get("file_type", "").lower()
    doc_type = metadata.get("document_type") or detect_document_type(text, metadata)

    if doc_type == "faq" or metadata.get("question"):
        return "atomic"

    if file_type in ["csv", "xlsx", "xlsm", "json", "jsonl", "xml"]:
        return "atomic"

    if doc_type == "announcement":
        return "atomic"
    if doc_type == "directory":
        return "directory_sections"

    if "## " in text:
        return "marked_sections"
    

    return "prose"


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
        doc_type = metadata.get("document_type") or detect_document_type(text, metadata)

        document.metadata.update(
            {
                **metadata,
                "chunk_strategy": strategy,
                "document_type": doc_type,
            }
        )

        if strategy == "directory_sections":
            prepared.extend(split_directory_entries(document))
        elif strategy == "marked_sections":
            prepared.extend(split_marked_sections(document))
        else:
            prepared.append(document)

    return prepared


def chunk_documents(documents: List[Document]) -> List[BaseNode]:
    prepared_documents = prepare_documents_for_chunking(documents)
    nodes: list[BaseNode] = []

    for document in prepared_documents:
        strategy = document.metadata.get("chunk_strategy", "prose")
        splitter = atomic_splitter if strategy == "atomic" else prose_splitter

        document_nodes = splitter.get_nodes_from_documents([document])

        for local_index, node in enumerate(document_nodes):
            content = node.get_content()
            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

            document_id = node.metadata["document_id"]
            version = node.metadata["document_version"]
            section_index = node.metadata.get("section_index", 0)

            node.id_ = (
                f"{document_id}:v{version}:"
                f"s{section_index}:c{local_index}:{digest}"
            )

            node.metadata.update(
                {
                    "chunk_id": node.id_,
                    "chunk_index": local_index,
                    "chunk_strategy": strategy,
                    "chunk_size": ATOMIC_MAX_TOKENS if strategy == "atomic" else PROSE_CHUNK_SIZE,
                    "chunk_overlap": 0 if strategy == "atomic" else PROSE_CHUNK_OVERLAP,
                }
            )

        nodes.extend(document_nodes)

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