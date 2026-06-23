import os

from dotenv import load_dotenv
from llama_index.vector_stores.postgres import PGVectorStore


load_dotenv()

DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
VECTOR_TABLE = os.getenv("VECTOR_TABLE", "hkpl_knowledge")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))

vector_store = PGVectorStore.from_params(
    database=os.getenv("POSTGRES_DB", "hkpl_vector_db"),
    user=os.getenv("POSTGRES_USER", "postgres"),
    password=DB_PASSWORD,
    host=os.getenv("POSTGRES_HOST", "postgres"),
    port=int(os.getenv("POSTGRES_PORT", "5432")),
    table_name=VECTOR_TABLE,
    embed_dim=EMBED_DIM,
)
