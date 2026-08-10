"""Shared SQLAlchemy connection for non-vector PostgreSQL operations.

The ingestion registry, corpus maintenance, compliance, and conversation
history code use this engine for ordinary SQL. Vector insertion and similarity
search use the separate LlamaIndex PGVectorStore configured in
``infrastructure.vector_store``.
"""

import os

from dotenv import load_dotenv
from sqlalchemy import create_engine


load_dotenv()

DB_URL = os.getenv(
    "DB_URL",
    "postgresql://postgres:postgres@postgres:5432/hkpl_vector_db",
)

engine = create_engine(
    DB_URL,
    pool_pre_ping=True,
    future=True,
)
