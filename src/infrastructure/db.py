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
