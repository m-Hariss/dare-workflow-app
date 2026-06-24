"""Stores and parses the active workflow JSON.

Persists to {data_dir}/workflow.json across restarts. Parses the workflow to
extract what it needs at runtime: LLM providers, files, embeddings, user_input.

Supports both camelCase (old Dare export) and snake_case (new export) field names.
"""
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Maps workflow provider names → key_store names
_PROVIDER_MAP = {
    "anthropic": "claude",
    "claude":    "claude",
    "openai":    "openai",
    "custom":    "openai",
    "google":    "gemini",
    "gemini":    "gemini",
    "ollama":    "ollama",
    "llama":     "ollama",
}


class WorkflowStore:
    def __init__(self, data_dir: Path):
        self._path = Path(data_dir) / "workflow.json"

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, payload: dict):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(payload, indent=2))
        logger.info("Saved workflow: %s", payload.get("title", "Untitled"))

    def load(self) -> dict | None:
        if not self._path.exists():
            return None
        try:
            return json.loads(self._path.read_text())
        except Exception as e:
            logger.error("Failed to load workflow: %s", e)
            return None

    def clear(self):
        if self._path.exists():
            self._path.unlink()

    def exists(self) -> bool:
        return self._path.exists()

    # ------------------------------------------------------------------
    # Info / parsing
    # ------------------------------------------------------------------

    def get_info(self) -> dict | None:
        payload = self.load()
        return self._parse(payload) if payload else None

    def _parse(self, payload: dict) -> dict:
        nodes = payload.get("nodes", [])

        required_providers = set()
        required_files     = []
        needs_files        = False
        needs_embeddings   = False
        has_user_input     = False
        steps              = []

        for node in nodes:
            ntype = node.get("type")
            data  = node.get("data", {})

            if ntype in ("step", "router"):
                # LLM provider
                llm         = data.get("llm") or {}
                raw_provider = (llm.get("provider") or "").lower()
                ks_provider  = _PROVIDER_MAP.get(raw_provider, raw_provider)
                if ks_provider:
                    required_providers.add(ks_provider)

                # Detect {{user_input}} in prompt or text input
                prompt_content = (data.get("prompt") or {}).get("content", "")
                text_input     = data.get("textInput") or data.get("text_input") or ""
                if "{{user_input}}" in prompt_content or "{{user_input}}" in text_input:
                    has_user_input = True

                # RAG files (camelCase + snake_case)
                rag             = data.get("rag", {})
                content_files   = rag.get("contentFiles")   or rag.get("content_files",   [])
                embedding_files = rag.get("embeddingFiles") or rag.get("embedding_files", [])

                for f in content_files:
                    name = f.get("name") or str(f.get("fileId") or f.get("file_id", ""))
                    if name:
                        needs_files = True
                        required_files.append(name)

                for f in embedding_files:
                    name = f.get("name") or str(f.get("fileId") or f.get("file_id", ""))
                    if name:
                        needs_files      = True
                        needs_embeddings = True
                        required_files.append(name)

                if ntype == "step":
                    steps.append({
                        "id":       node.get("id"),
                        "label":    data.get("label", "Step"),
                        "prompt":   prompt_content,
                        "provider": ks_provider,
                        "model":    llm.get("identifier", ""),
                    })

            elif ntype == "file":
                mode  = data.get("retrievalMode") or data.get("retrieval_mode", "content")
                for f in data.get("files", []):
                    name = f.get("name") or str(f.get("fileId") or f.get("file_id", ""))
                    if name:
                        needs_files = True
                        required_files.append(name)
                if mode in ("embeddings", "both"):
                    needs_embeddings = True

        return {
            "title":              payload.get("title", "Untitled Workflow"),
            "description":        payload.get("description", ""),
            "mode":               payload.get("mode", "sequential"),
            "steps":              steps,
            "required_providers": list(required_providers),
            "required_files":     list(dict.fromkeys(required_files)),
            "needs_files":        needs_files,
            "needs_embeddings":   needs_embeddings,
            "has_user_input":     has_user_input,
            "node_count":         len(nodes),
        }
