"""Bundles the service clients handlers depend on.

Everything runs locally — no external API calls except to the user's chosen
LLM provider. Files are read from the local files folder; embeddings and
retrieval use ChromaDB on disk.
"""
from pathlib import Path
from typing import Optional

from providers.embeddings import EmbeddingClient
from providers.llm import LLMClient
from rag.pipeline import EmbeddingPipeline
from storage.file_store import FileStore
from storage.vector_store import VectorStore

_DEFAULT_EMBED_PROVIDER = "openai"
_DEFAULT_EMBED_MODEL    = "text-embedding-3-small"


class Services:
    def __init__(
        self,
        llm: LLMClient,
        file_store: FileStore,
        embed: EmbeddingPipeline,
    ):
        self.llm        = llm
        self.file_store = file_store
        self.embed      = embed

    @classmethod
    def from_config(cls, key_store=None, data_dir: Path | None = None, file_map: dict | None = None):
        api_keys: dict             = {}
        ollama_host: Optional[str] = None
        files_folder: Optional[str] = None
        embed_provider = _DEFAULT_EMBED_PROVIDER
        embed_model    = _DEFAULT_EMBED_MODEL

        if key_store is not None:
            for provider in ("openai", "claude", "gemini"):
                v = key_store.get(provider)
                if v:
                    api_keys[provider] = v
            ollama_host    = key_store.get("ollama")
            files_folder   = key_store.get("files_folder")
            embed_provider = key_store.get("embed_provider") or _DEFAULT_EMBED_PROVIDER
            embed_model    = key_store.get("embed_model")    or _DEFAULT_EMBED_MODEL

        llm        = LLMClient(api_keys=api_keys or None, ollama_host=ollama_host)
        file_store = FileStore(files_folder, file_map=file_map)

        effective_data_dir = data_dir or Path(".local_data")
        embed_api_key = api_keys.get(embed_provider) or api_keys.get("openai")
        embed_client = EmbeddingClient(
            provider=embed_provider,
            model=embed_model,
            api_key=embed_api_key,
            ollama_host=ollama_host,
        )
        embed_pipeline = EmbeddingPipeline(
            vector_store=VectorStore(effective_data_dir),
            embed_client=embed_client,
            data_dir=effective_data_dir,
        )

        return cls(llm=llm, file_store=file_store, embed=embed_pipeline)
