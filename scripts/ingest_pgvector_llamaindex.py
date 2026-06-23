#!/usr/bin/env python3

import csv
import os
from pathlib import Path
from uuid import uuid5, NAMESPACE_URL

from llama_index.core import (
    Document,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.vector_stores import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)

from src.infrastructure.embedding import (
    embed_model,
)
from src.infrastructure.vector_store import (
    VECTOR_TABLE,
    vector_store,
)
from src.ingestion.chunking import (
    chunk_documents,
)


def load_faq_documents(
    csv_path: str,
) -> list[Document]:
    documents: list[Document] = []

    with open(
        csv_path,
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        for row_index, row in enumerate(reader):
            question = row.get(
                "query",
                "",
            ).strip()
            answer = row.get(
                "expected_answer_text",
                "",
            ).strip()

            if not question or not answer:
                continue

            source_url = row.get(
                "source_url",
                "https://www.hkpl.gov.hk/en/ask-a-librarian/faq.html",
            ).strip()
            row_id = row.get(
                "source_row_id",
                str(row_index),
            ).strip()

            document_id = str(
                uuid5(
                    NAMESPACE_URL,
                    f"{source_url}#faq-{row_id}",
                )
            )

            document = Document(
                text=(
                    f"Question: {question}\n"
                    f"Answer: {answer}"
                ),
                metadata={
                    "document_id": document_id,
                    "original_file_name": (
                        Path(csv_path).name
                    ),
                    "file_name": (
                        Path(csv_path).name
                    ),
                    "file_type": "csv",
                    "source_title": row.get(
                        "source_title",
                        "HKPL Ask a Librarian FAQ",
                    ).strip(),
                    "source": row.get(
                        "source_title",
                        "HKPL Ask a Librarian FAQ",
                    ).strip(),
                    "source_url": source_url,
                    "url": source_url,
                    "source_type": row.get(
                        "source_type",
                        "official_website",
                    ).strip(),
                    "access_level": "public",
                    "document_version": 1,
                    "domain": row.get(
                        "domain",
                        "",
                    ).strip(),
                    "question": question,
                    "snippet": row.get(
                        "expected_context_snippet",
                        "",
                    ).strip(),
                    "row_id": row_id,
                    "row_number": row_index + 2,
                    "section_index": row_index,
                    "chunk_strategy": "atomic",
                },
            )
            document.id_ = (
                f"{document_id}:v1:section:0"
            )
            documents.append(document)

    return documents



def delete_existing_faq_chunks() -> int:
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="source_title",
                value="HKPL Ask a Librarian FAQ",
                operator=FilterOperator.EQ,
            )
        ]
    )
    nodes = vector_store.get_nodes(filters=filters)
    if not nodes:
        return 0

    vector_store.delete_nodes(
        node_ids=[node.node_id for node in nodes]
    )
    return len(nodes)


def main() -> None:
    data_path = os.getenv(
        "DATA_PATH",
        "/app/data/hkpl_faq_clean.csv",
    )
    rebuild_all = (
        os.getenv(
            "REBUILD_ALL",
            "false",
        ).lower()
        == "true"
    )

    if rebuild_all:
        print(
            "REBUILD_ALL=true: clearing "
            f"data_{VECTOR_TABLE}"
        )
        vector_store.clear()

    removed = delete_existing_faq_chunks()
    print(f"Removed {removed} existing FAQ chunks")

    documents = load_faq_documents(
        data_path
    )
    print(
        f"Loaded {len(documents)} FAQ documents"
    )

    nodes = chunk_documents(
        documents
    )
    print(
        f"Created {len(nodes)} FAQ chunks"
    )

    storage_context = (
        StorageContext.from_defaults(
            vector_store=vector_store
        )
    )

    VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )

    print(
        "Ingested FAQ data into "
        f"data_{VECTOR_TABLE}"
    )


if __name__ == "__main__":
    main()
