"""Shared application state and helpers.

Holds the singletons (data dir, key/workflow stores) and small helpers that the
API routers depend on. Kept separate from main.py so routers can import these
without pulling in the FastAPI app or each other.
"""
import json
import logging
from pathlib import Path

from dotenv import load_dotenv
from syft_core import Client

from clients.llm_client import LLMClient
from embedding_pipeline import EmbeddingPipeline
from key_store import KeyStore
from vector_store import VectorStore
from workflow_store import WorkflowStore

logger = logging.getLogger(__name__)

load_dotenv(Path(__file__).resolve().parent / ".env")

app_name = Path(__file__).resolve().parent.name


# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------

def _init_data_dir() -> Path:
    try:
        client = Client.load()
        return Path(client.app_data(app_name))
    except Exception:
        return Path(__file__).resolve().parent / ".local_data"


DATA_DIR      = _init_data_dir()
UPLOADS_DIR   = DATA_DIR / "uploads"
FILE_MAP_PATH = DATA_DIR / "file_map.json"

key_store      = KeyStore(DATA_DIR)
workflow_store = WorkflowStore(DATA_DIR)


# ---------------------------------------------------------------------------
# File-slot mapping helpers
# ---------------------------------------------------------------------------

def load_file_map() -> dict:
    if FILE_MAP_PATH.exists():
        try:
            return json.loads(FILE_MAP_PATH.read_text())
        except Exception:
            return {}
    return {}


def save_file_map(fmap: dict):
    FILE_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    FILE_MAP_PATH.write_text(json.dumps(fmap, indent=2))


# ---------------------------------------------------------------------------
# Embedding pipeline factory
# ---------------------------------------------------------------------------

def make_embed_pipeline() -> EmbeddingPipeline:
    provider = key_store.get("embed_provider") or "openai"
    model    = key_store.get("embed_model")    or "text-embedding-3-small"
    api_keys = {p: v for p in ("openai", "claude", "gemini") if (v := key_store.get(p))}
    llm = LLMClient(api_keys=api_keys or None, ollama_host=key_store.get("ollama"))
    return EmbeddingPipeline(
        vector_store=VectorStore(DATA_DIR),
        llm_client=llm,
        data_dir=DATA_DIR,
        provider=provider,
        model=model,
    )
