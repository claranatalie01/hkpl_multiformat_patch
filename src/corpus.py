from __future__ import annotations

from collections.abc import Iterable

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import BaseNode
from sqlalchemy import text

from .infrastructure.db import engine
from .infrastructure.embedding import embed_model
from .infrastructure.vector_store import VECTOR_TABLE, vector_store


PRIMARY_DATASET = "hkpl"
PRIMARY_CORPUS_ROLE = "primary"
DISTRACTOR_CORPUS_ROLE = "distractor"
LEGACY_DISTRACTOR_DATASETS = frozenset({"hotpotqa", "webz_news"})
VECTOR_TABLE_NAME = f"data_{VECTOR_TABLE}"


def is_distractor_metadata(metadata: dict | None) -> bool:
    metadata = metadata or {}
    role = str(metadata.get("corpus_role") or "").strip().lower()
    dataset = str(metadata.get("dataset") or "").strip().lower()
    return (
        role == DISTRACTOR_CORPUS_ROLE
        or dataset in LEGACY_DISTRACTOR_DATASETS
    )


def normalize_corpus_roles() -> None:
    """Backfill corpus labels without changing text or embeddings."""
    with engine.begin() as connection:
        connection.execute(
            text(f"""
                UPDATE {VECTOR_TABLE_NAME}
                SET metadata_ = jsonb_set(
                    metadata_,
                    '{{dataset}}',
                    to_jsonb(CAST(:primary_dataset AS text)),
                    true
                )
                WHERE COALESCE(metadata_->>'dataset', '') = ''
            """),
            {"primary_dataset": PRIMARY_DATASET},
        )
        connection.execute(
            text(f"""
                UPDATE {VECTOR_TABLE_NAME}
                SET metadata_ = jsonb_set(
                    metadata_,
                    '{{corpus_role}}',
                    to_jsonb(CAST(:distractor_role AS text)),
                    true
                )
                WHERE metadata_->>'dataset' = ANY(CAST(:datasets AS text[]))
                  AND COALESCE(metadata_->>'corpus_role', '')
                      IS DISTINCT FROM :distractor_role
            """),
            {
                "datasets": list(LEGACY_DISTRACTOR_DATASETS),
                "distractor_role": DISTRACTOR_CORPUS_ROLE,
            },
        )
        connection.execute(
            text(f"""
                UPDATE {VECTOR_TABLE_NAME}
                SET metadata_ = jsonb_set(
                    metadata_,
                    '{{corpus_role}}',
                    to_jsonb(CAST(:primary_role AS text)),
                    true
                )
                WHERE COALESCE(metadata_->>'corpus_role', '') = ''
                  AND metadata_->>'dataset'
                      <> ALL(CAST(:distractor_datasets AS text[]))
            """),
            {
                "primary_role": PRIMARY_CORPUS_ROLE,
                "distractor_datasets": list(LEGACY_DISTRACTOR_DATASETS),
            },
        )


def replace_dataset_vectors(dataset: str, nodes: Iterable[BaseNode]) -> int:
    dataset = dataset.strip().lower()
    if not dataset or dataset == PRIMARY_DATASET:
        raise ValueError("A non-primary dataset name is required.")

    materialized_nodes = list(nodes)
    with engine.begin() as connection:
        deleted = connection.execute(
            text(f"""
                DELETE FROM {VECTOR_TABLE_NAME}
                WHERE metadata_->>'dataset' = :dataset
            """),
            {"dataset": dataset},
        )

    if materialized_nodes:
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        VectorStoreIndex(
            materialized_nodes,
            storage_context=storage_context,
            embed_model=embed_model,
            show_progress=True,
        )
    normalize_corpus_roles()
    return int(deleted.rowcount or 0)
