"""HTTP client for Dare's data APIs.

The runner delegates all file access and retrieval to Dare; it performs no
embedding, indexing, or storage itself.
"""
import requests


class DareClient:
    def __init__(self, base_url: str, token: str, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        if token:
            self.session.headers.update({"Authorization": f"Bearer {token}"})

    def export_workflow(self, workflow_id: int) -> dict:
        url = f"{self.base_url}/api/workflows/{workflow_id}/export/"
        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def get_file_content(self, file_id: int) -> str:
        url = f"{self.base_url}/api/files/{file_id}/content/"
        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json().get("content", "")

    def retrieval_query(self, query, file_ids, top_k=10, similarity_threshold=0.5) -> list:
        url = f"{self.base_url}/api/retrieval/query/"
        payload = {
            "query": query,
            "file_ids": file_ids,
            "top_k": top_k,
            "similarity_threshold": similarity_threshold,
        }
        resp = self.session.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        chunks = resp.json().get("chunks", [])
        # Normalize camelCase response keys -> snake_case for handlers.
        return [
            {
                "text": c.get("text", ""),
                "score": c.get("score", 0.0),
                "file_id": c.get("fileId", c.get("file_id")),
                "file_name": c.get("fileName", c.get("file_name", "Unknown file")),
                "chunk_index": c.get("chunkIndex", c.get("chunk_index", 0)),
            }
            for c in chunks
        ]
