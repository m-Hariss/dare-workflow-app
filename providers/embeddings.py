"""Embedding provider client.

Handles vector embedding calls only (separate from chat completions).
Supports OpenAI and Ollama embedding endpoints.
"""
import logging
import os

import requests

logger = logging.getLogger(__name__)

OPENAI_DEFAULT_BASE = "https://api.openai.com/v1"


class EmbeddingClient:
    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str | None = None,
        ollama_host: str | None = None,
        timeout: int = 60,
    ):
        self.provider = provider.lower()
        self.model = model
        self._api_key = api_key
        self._ollama_host = ollama_host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self._timeout = timeout
        logger.info("[EmbeddingClient] provider=%s  model=%s  key_set=%s",
                    self.provider, self.model, bool(self._api_key))

    def embed(self, text: str) -> list[float]:
        """Return a float embedding vector for the given text."""
        logger.debug("[EmbeddingClient.embed] provider=%s  text_len=%d", self.provider, len(text))
        if self.provider == "openai":
            return self._openai_embed(text)
        if self.provider == "ollama":
            return self._ollama_embed(text)
        raise ValueError(
            f"Unsupported embedding provider: {self.provider!r}. Use 'openai' or 'ollama'."
        )

    def _openai_embed(self, text: str) -> list[float]:
        key = self._api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            logger.error("[EmbeddingClient] No OpenAI API key — set it in Step 4 or as OPENAI_API_KEY env var")
            raise RuntimeError("No OpenAI API key configured for embeddings.")
        logger.debug("[EmbeddingClient] POST openai/embeddings  model=%s  text_len=%d", self.model, len(text))
        resp = requests.post(
            f"{OPENAI_DEFAULT_BASE}/embeddings",
            json={"input": text, "model": self.model},
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            timeout=self._timeout,
        )
        if not resp.ok:
            logger.error("[EmbeddingClient] OpenAI embedding failed  status=%d  body=%s",
                         resp.status_code, resp.text[:500])
            raise RuntimeError(
                f"Embedding call failed ({resp.status_code}) for model '{self.model}': {resp.text}"
            )
        vec = resp.json()["data"][0]["embedding"]
        logger.debug("[EmbeddingClient] OpenAI embedding OK  dims=%d", len(vec))
        return vec

    def _ollama_embed(self, text: str) -> list[float]:
        url = f"{self._ollama_host.rstrip('/')}/api/embeddings"
        logger.debug("[EmbeddingClient] POST ollama/embeddings  url=%s  model=%s  text_len=%d",
                     url, self.model, len(text))
        resp = requests.post(
            url,
            json={"model": self.model, "prompt": text},
            headers={"Content-Type": "application/json"},
            timeout=self._timeout,
        )
        if not resp.ok:
            logger.error("[EmbeddingClient] Ollama embedding failed  status=%d  body=%s",
                         resp.status_code, resp.text[:500])
            raise RuntimeError(
                f"Embedding call failed ({resp.status_code}) for model '{self.model}': {resp.text}"
            )
        vec = resp.json()["embedding"]
        logger.debug("[EmbeddingClient] Ollama embedding OK  dims=%d", len(vec))
        return vec
