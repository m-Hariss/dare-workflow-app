"""App-level singletons and service factory.

Single import point for the API layer — replaces the old deps.py + services.py pair.
"""
import json
import logging
import threading
from pathlib import Path

from dotenv import load_dotenv
from syft_core import Client

from providers.embeddings import EmbeddingClient
from providers.llm import LLMClient
from rag.pipeline import EmbeddingPipeline
from storage.file_store import FileStore
from storage.key_store import KeyStore
from storage.vector_store import VectorStore
from storage.workflow_store import WorkflowStore

logger = logging.getLogger(__name__)

load_dotenv(Path(__file__).resolve().parent / ".env")

app_name = Path(__file__).resolve().parent.name


# ---------------------------------------------------------------------------
# Singletons — created once at startup
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

# In-memory embed progress — written from the embed thread, read by the progress endpoint.
# { slot_id: {"current": 3, "total": 10, "done": False, "file": "name"} }
_embed_progress: dict = {}


def set_embed_progress(slot_id: str, current: int, total: int, done: bool = False, file: str = ""):
    _embed_progress[slot_id] = {"current": current, "total": total, "done": done, "file": file}


def get_embed_progress(slot_id: str) -> dict:
    return _embed_progress.get(slot_id, {"current": 0, "total": 0, "done": False, "file": ""})


def clear_embed_progress(slot_id: str):
    _embed_progress.pop(slot_id, None)


# Shared singleton — one ChromaDB PersistentClient per process.
# _chroma_lock serialises all ChromaDB writes so concurrent embed requests
# (one per slot) don't race on the same SQLite file.
_shared_vector_store: "VectorStore | None" = None
_chroma_lock = threading.Lock()


def _get_vector_store() -> VectorStore:
    global _shared_vector_store
    if _shared_vector_store is None:
        _shared_vector_store = VectorStore(DATA_DIR)
    return _shared_vector_store


# ---------------------------------------------------------------------------
# File-slot mapping
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
# Services — assembled fresh per request/run from current config
# ---------------------------------------------------------------------------

class Services:
    """Bundles the three things the execution engine needs for one run."""
    def __init__(self, llm: LLMClient, file_store: FileStore, embed: EmbeddingPipeline):
        self.llm        = llm
        self.file_store = file_store
        self.embed      = embed


def _build_llm_client() -> LLMClient:
    api_keys = {p: v for p in ("openai", "claude", "gemini") if (v := key_store.get(p))}
    return LLMClient(api_keys=api_keys or None, ollama_host=key_store.get("ollama"))


def _build_embed_pipeline() -> EmbeddingPipeline:
    # Embedding model is always text-embedding-3-large via OpenAI (matches Dare's OpenAIWrapper).
    client = EmbeddingClient(
        provider="openai",
        model="text-embedding-3-large",
        api_key=key_store.get("openai"),
        ollama_host=key_store.get("ollama"),
    )
    return EmbeddingPipeline(_get_vector_store(), client, DATA_DIR)


def make_services() -> Services:
    """Build the service bundle for one workflow run."""
    return Services(
        llm=_build_llm_client(),
        file_store=FileStore(key_store.get("files_folder"), file_map=load_file_map()),
        embed=_build_embed_pipeline(),
    )


def make_embed_pipeline() -> EmbeddingPipeline:
    """Build embedding pipeline — used by the /index endpoint."""
    return _build_embed_pipeline()


# ---------------------------------------------------------------------------
# Slot embedding helpers
# ---------------------------------------------------------------------------

def embed_slot(slot_id: str) -> int:
    """Chunk + embed an uploaded slot file into ChromaDB. Returns chunk count."""
    logger.info("[embed_slot] START  slot=%s", slot_id)
    set_embed_progress(slot_id, 0, 0, done=False)
    try:
        store   = FileStore(key_store.get("files_folder"), file_map=load_file_map())
        content = store.get_content(slot_id)
        logger.info("[embed_slot] content_len=%d  slot=%s", len(content), slot_id)

        def on_progress(current: int, total: int, filename: str):
            set_embed_progress(slot_id, current, total, done=False, file=filename)

        if not _chroma_lock.acquire(timeout=120):
            raise RuntimeError("Timed out waiting for embedding lock — another embed may still be running.")
        try:
            chunks = _build_embed_pipeline().index_file(slot_id, content, on_progress=on_progress)
        finally:
            _chroma_lock.release()

        set_embed_progress(slot_id, chunks, chunks, done=True)
        return chunks
    except Exception:

        set_embed_progress(slot_id, 0, 0, done=True)
        raise


def remove_slot_embeddings(slot_id: str):
    """Drop indexed chunks + status for a slot (best effort)."""
    try:
        with _chroma_lock:
            pipeline = _build_embed_pipeline()
            pipeline._store.delete_by_filename(slot_id)
            pipeline.remove_status(slot_id)
    except Exception as e:
        logger.warning("Could not remove embeddings for slot %s: %s", slot_id, e)


def read_index_status() -> dict:
    p = DATA_DIR / "index_status.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}
