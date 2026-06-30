"""Embedding pipeline — chunks documents, generates embeddings, manages index status.

Chunking uses a paragraph-aware splitter (tries \n\n, then \n, then ". ") so
chunks don't break mid-sentence when possible.

Index status is persisted to {data_dir}/index_status.json so the dashboard can
show which files are indexed and which model was used.
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_CHUNK_SIZE = 1000   # characters
_DEFAULT_OVERLAP    = 200


def chunk_text(text: str, chunk_size: int = _DEFAULT_CHUNK_SIZE, overlap: int = _DEFAULT_OVERLAP) -> list[str]:
    """Split text into overlapping chunks, preferring natural break points."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            # Prefer to break at a paragraph, then line, then sentence boundary.
            for sep in ("\n\n", "\n", ". "):
                pos = text.rfind(sep, start, end)
                if pos != -1:
                    end = pos + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap

    return chunks


class EmbeddingPipeline:
    def __init__(self, vector_store, embed_client, data_dir: Path):
        self._store  = vector_store
        self._client = embed_client
        self._status_file = Path(data_dir) / "index_status.json"
        # Expose provider/model for status reporting (sourced from the embed client).
        self.provider = embed_client.provider
        self.model    = embed_client.model

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_file(self, filename: str, content: str) -> int:
        """Chunk, embed, and store one file. Returns number of chunks indexed."""
        logger.info("[pipeline] index_file START  file=%s  content_len=%d", filename, len(content))

        chunks = chunk_text(content)
        if not chunks:
            logger.warning("[pipeline] No chunks produced for %s — content may be empty or whitespace-only", filename)
            return 0

        logger.info("[pipeline] Chunked into %d chunks  file=%s", len(chunks), filename)
        self._store.delete_by_filename(filename)

        embeddings = []
        for i, chunk in enumerate(chunks):
            logger.info("[pipeline] Embedding chunk %d/%d  file=%s  provider=%s  model=%s",
                        i + 1, len(chunks), filename, self._client.provider, self._client.model)
            try:
                emb = self._client.embed(chunk)
            except Exception as e:
                logger.error("[pipeline] Embedding FAILED at chunk %d/%d  file=%s  error=%s",
                             i + 1, len(chunks), filename, e)
                raise
            embeddings.append(emb)

        logger.info("[pipeline] All chunks embedded, writing to vector store  file=%s  chunks=%d",
                    filename, len(chunks))
        self._store.upsert_chunks(filename, chunks, embeddings)
        self._record_status(filename, len(chunks))
        logger.info("[pipeline] index_file DONE  file=%s  chunks=%d", filename, len(chunks))
        return len(chunks)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 10,
        threshold: float = 0.5,
        filenames: list[str] | None = None,
    ) -> list[dict]:
        """Embed query and return matching chunks from ChromaDB."""
        query_embedding = self._client.embed(query)
        return self._store.query(query_embedding, top_k, threshold, filenames)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        if not self._status_file.exists():
            return {}
        try:
            return json.loads(self._status_file.read_text())
        except Exception:
            return {}

    def remove_status(self, filename: str):
        status = self.get_status()
        status.pop(filename, None)
        self._write_status(status)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _record_status(self, filename: str, chunks: int):
        status = self.get_status()
        status[filename] = {
            "indexed_at": datetime.now(timezone.utc).isoformat(),
            "chunks": chunks,
            "provider": self.provider,
            "model": self.model,
        }
        self._write_status(status)

    def _write_status(self, status: dict):
        self._status_file.write_text(json.dumps(status, indent=2))
