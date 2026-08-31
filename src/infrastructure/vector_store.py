"""Configure the LlamaIndex-backed PostgreSQL vector collection.

``postgres-init/init.sql`` prepares PostgreSQL and enables the pgvector
extension. This module supplies the application-level connection details,
collection name, and embedding dimension. LlamaIndex creates/manages the
physical ``data_<VECTOR_TABLE>`` table when the vector store is used.
"""

import os

from dotenv import load_dotenv
from llama_index.vector_stores.postgres import PGVectorStore


load_dotenv()

DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
VECTOR_TABLE = os.getenv("VECTOR_TABLE", "hkpl_knowledge")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))

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
)
