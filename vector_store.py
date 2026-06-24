"""ChromaDB wrapper for local vector storage.

One persistent collection ("workflow_files") holds all indexed chunks from all
files in the files folder.  Chunk IDs are "{filename}::{index}" so we can
delete and re-index individual files without touching the rest.

Cosine similarity space is used so that scores map naturally to [0, 1].
"""
import logging
from pathlib import Path

import chromadb

logger = logging.getLogger(__name__)

_COLLECTION = "workflow_files"


class VectorStore:
    def __init__(self, data_dir: Path):
        db_path = Path(data_dir) / "chroma_db"
        db_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(db_path))
        # cosine space: distance = 1 - cosine_similarity → score = 1 - distance
        self._col = self._client.get_or_create_collection(
            _COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert_chunks(self, filename: str, chunks: list[str], embeddings: list[list[float]]):
        if not chunks:
            return
        ids = [f"{filename}::{i}" for i in range(len(chunks))]
        metadatas = [{"filename": filename, "chunk_index": i} for i in range(len(chunks))]
        self._col.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info("Upserted %d chunks for %s", len(chunks), filename)

    def delete_by_filename(self, filename: str):
        try:
            self._col.delete(where={"filename": filename})
        except Exception as e:
            # ChromaDB raises if no documents match — safe to ignore
            logger.debug("delete_by_filename(%s) no-op: %s", filename, e)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        threshold: float = 0.5,
        filenames: list[str] | None = None,
    ) -> list[dict]:
        """Return chunks whose cosine similarity to query_embedding ≥ threshold."""
        total = self._col.count()
        if total == 0:
            return []

        where = {"filename": {"$in": filenames}} if filenames else None
        n = min(top_k, total)

        try:
            results = self._col.query(
                query_embeddings=[query_embedding],
                n_results=n,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error("ChromaDB query failed: %s", e)
            return []

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        chunks = []
        for doc, meta, dist in zip(docs, metas, dists):
            score = max(0.0, 1.0 - dist)   # cosine distance → similarity
            if score >= threshold:
                chunks.append({
                    "text": doc,
                    "score": round(score, 4),
                    "filename": meta.get("filename", ""),
                    "chunk_index": meta.get("chunk_index", 0),
                })

        return sorted(chunks, key=lambda c: c["score"], reverse=True)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def count(self) -> int:
        return self._col.count()
