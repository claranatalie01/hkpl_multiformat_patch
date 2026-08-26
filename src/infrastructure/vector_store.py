"""Configure the LlamaIndex-backed PostgreSQL vector collection.

``postgres-init/init.sql`` prepares PostgreSQL and enables the pgvector
extension. This module supplies the application-level connection details,
collection name, and embedding dimension. LlamaIndex creates/manages the
physical ``data_<VECTOR_TABLE>`` table when the vector store is used.
"""

import os
import re
import threading

from dotenv import load_dotenv
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy import text

from .db import engine


load_dotenv()

DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
VECTOR_TABLE = os.getenv("VECTOR_TABLE", "hkpl_knowledge").strip().lower()
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))
TEXT_SEARCH_CONFIG = os.getenv("TEXT_SEARCH_CONFIG", "english")
VECTOR_TABLE_NAME = f"data_{VECTOR_TABLE}"

if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", VECTOR_TABLE_NAME):
    raise ValueError(f"Unsafe vector table name: {VECTOR_TABLE_NAME!r}")

# This object is shared by ingestion (writes) and retrieval (similarity reads),
# ensuring both sides use the same collection and embedding dimension.
vector_store = PGVectorStore.from_params(
    database=os.getenv("POSTGRES_DB", "hkpl_vector_db"),
    user=os.getenv("POSTGRES_USER", "postgres"),
    password=DB_PASSWORD,
    host=os.getenv("POSTGRES_HOST", "postgres"),
    port=int(os.getenv("POSTGRES_PORT", "5432")),
    table_name=VECTOR_TABLE,
    embed_dim=EMBED_DIM,
    hybrid_search=True,
    text_search_config=TEXT_SEARCH_CONFIG,
)


_hybrid_schema_ready = False
_hybrid_schema_lock = threading.Lock()


def ensure_hybrid_search_schema() -> None:
    """Create the table and its trigram index once per process.

    PGVectorStore creates the full-text column and GIN index for a new table.
    Existing pre-hybrid tables are rejected with a clear rebuild instruction;
    adding the generated column in place would silently mix incompatible runs.
    """
    global _hybrid_schema_ready
    if _hybrid_schema_ready:
        return
    with _hybrid_schema_lock:
        if _hybrid_schema_ready:
            return

        # PGVectorStore has no public initialize-only method. Its own query and
        # insertion paths call this same idempotent initializer.
        vector_store._initialize()
        with engine.begin() as connection:
            has_text_search = connection.execute(
                text("""
                    SELECT EXISTS (
                        SELECT 1
                        FROM information_schema.columns
                        WHERE table_schema = 'public'
                          AND table_name = :table_name
                          AND column_name = 'text_search_tsv'
                    )
                """),
                {"table_name": VECTOR_TABLE_NAME},
            ).scalar_one()
            if not has_text_search:
                raise RuntimeError(
                    f"{VECTOR_TABLE_NAME} predates hybrid search. Use a fresh "
                    "VECTOR_TABLE or recreate and reingest it."
                )
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
            connection.execute(text(
                f"CREATE INDEX IF NOT EXISTS {VECTOR_TABLE_NAME}_text_trgm_idx "
                f"ON {VECTOR_TABLE_NAME} USING gin (lower(text) gin_trgm_ops)"
            ))
        _hybrid_schema_ready = True
