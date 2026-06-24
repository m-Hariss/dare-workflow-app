"""Bundles the service clients handlers depend on.

DareClient is retained only for workflow export (fetching the graph JSON by ID).
All file access goes through FileStore (local folder on disk).
Embeddings and semantic search go through EmbeddingPipeline (ChromaDB, local).
"""
import os
from pathlib import Path
from typing import Optional

from clients.dare_client import DareClient
from clients.llm_client import LLMClient
from embedding_pipeline import EmbeddingPipeline
from file_store import FileStore
from vector_store import VectorStore

_DEFAULT_EMBED_PROVIDER = "openai"
_DEFAULT_EMBED_MODEL    = "text-embedding-3-small"


class Services:
    def __init__(
        self,
        dare: DareClient,
        llm: LLMClient,
        file_store: FileStore,
        embed: EmbeddingPipeline,
    ):
        self.dare       = dare        # workflow export only — no file/retrieval calls
        self.llm        = llm
        self.file_store = file_store
        self.embed      = embed

    @classmethod
    def from_config(
        cls,
        base_url: str | None = None,
        token: str | None = None,
        key_store=None,
        data_dir: Path | None = None,
    ):
        base_url = base_url or os.environ.get("DARE_BASE_URL", "http://localhost:8000")
        token    = token    or os.environ.get("DARE_TOKEN", "")

        api_keys: dict       = {}
        ollama_host: Optional[str] = None
        files_folder: Optional[str] = None
        embed_provider = _DEFAULT_EMBED_PROVIDER
        embed_model    = _DEFAULT_EMBED_MODEL

        if key_store is not None:
            for provider in ("openai", "claude", "gemini"):
                v = key_store.get(provider)
                if v:
                    api_keys[provider] = v
            ollama_host   = key_store.get("ollama")
            files_folder  = key_store.get("files_folder")
            embed_provider = key_store.get("embed_provider") or _DEFAULT_EMBED_PROVIDER
            embed_model    = key_store.get("embed_model")    or _DEFAULT_EMBED_MODEL

        llm        = LLMClient(api_keys=api_keys or None, ollama_host=ollama_host)
        file_store = FileStore(files_folder)

        # EmbeddingPipeline requires a data_dir for ChromaDB + status file.
        # Falls back to a local .local_data/ dir when not running inside SyftBox.
        effective_data_dir = data_dir or Path(".local_data")
        vector_store  = VectorStore(effective_data_dir)
        embed_pipeline = EmbeddingPipeline(
            vector_store=vector_store,
            llm_client=llm,
            data_dir=effective_data_dir,
            provider=embed_provider,
            model=embed_model,
        )

        return cls(
            dare=DareClient(base_url, token),
            llm=llm,
            file_store=file_store,
            embed=embed_pipeline,
        )
