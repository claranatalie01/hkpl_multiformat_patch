import hashlib
import os
from typing import List

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode


PROSE_CHUNK_SIZE = int(
    os.getenv("CHUNK_SIZE", "512")
)
PROSE_CHUNK_OVERLAP = int(
    os.getenv("CHUNK_OVERLAP", "64")
)
ATOMIC_MAX_TOKENS = int(
    os.getenv("ATOMIC_MAX_TOKENS", "2048")
)

prose_splitter = SentenceSplitter(
    chunk_size=PROSE_CHUNK_SIZE,
    chunk_overlap=PROSE_CHUNK_OVERLAP,
)

atomic_splitter = SentenceSplitter(
    chunk_size=ATOMIC_MAX_TOKENS,
    chunk_overlap=0,
)


def chunk_documents(
    documents: List[Document],
) -> List[BaseNode]:
    nodes: list[BaseNode] = []

    for document in documents:
        strategy = document.metadata.get(
            "chunk_strategy",
            "prose",
        )

        splitter = (
            atomic_splitter
            if strategy == "atomic"
            else prose_splitter
        )

        document_nodes = splitter.get_nodes_from_documents(
            [document]
        )

        for local_index, node in enumerate(document_nodes):
            content = node.get_content()
            digest = hashlib.sha256(
                content.encode("utf-8")
            ).hexdigest()[:16]

            document_id = node.metadata["document_id"]
            version = node.metadata["document_version"]
            section_index = node.metadata.get(
                "section_index",
                0,
            )

            node.id_ = (
                f"{document_id}:v{version}:"
                f"s{section_index}:c{local_index}:{digest}"
            )

            node.metadata.update(
                {
                    "chunk_id": node.id_,
                    "chunk_index": local_index,
                    "chunk_strategy": strategy,
                    "chunk_size": (
                        ATOMIC_MAX_TOKENS
                        if strategy == "atomic"
                        else PROSE_CHUNK_SIZE
                    ),
                    "chunk_overlap": (
                        0
                        if strategy == "atomic"
                        else PROSE_CHUNK_OVERLAP
                    ),
                }
            )

        nodes.extend(document_nodes)

    return nodes
