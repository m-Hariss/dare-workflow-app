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

from providers.embeddings import EmbeddingClient
from rag.pipeline import EmbeddingPipeline
from storage.file_store import FileStore
from storage.key_store import KeyStore
from storage.vector_store import VectorStore
from storage.workflow_store import WorkflowStore

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
    api_key  = key_store.get(provider) or key_store.get("openai")
    embed_client = EmbeddingClient(
        provider=provider,
        model=model,
        api_key=api_key,
        ollama_host=key_store.get("ollama"),
    )
    return EmbeddingPipeline(
        vector_store=VectorStore(DATA_DIR),
        embed_client=embed_client,
        data_dir=DATA_DIR,
    )


# ---------------------------------------------------------------------------
# Slot embedding — chunk+embed an uploaded slot file into ChromaDB under its
# slot id, which is exactly the filename the file handler searches at run time.
# ---------------------------------------------------------------------------

def embed_slot(slot_id: str) -> int:
    """Index the file currently mapped to slot_id. Returns chunk count.

    Raises if no file is mapped, the embedding provider/key is unavailable, or
    the file can't be read — callers surface that to the user.
    """
    store   = FileStore(key_store.get("files_folder"), file_map=load_file_map())
    content = store.get_content(slot_id)
    return make_embed_pipeline().index_file(slot_id, content)


def remove_slot_embeddings(slot_id: str):
    """Drop any indexed chunks + status for a slot (best effort)."""
    try:
        pipeline = make_embed_pipeline()
        pipeline._store.delete_by_filename(slot_id)
        pipeline.remove_status(slot_id)
    except Exception as e:
        logger.warning("Could not remove embeddings for slot %s: %s", slot_id, e)


def read_index_status() -> dict:
    """Per-file index status, read straight from disk (no ChromaDB needed)."""
    p = DATA_DIR / "index_status.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}
