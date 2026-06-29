"""Stores and parses the active workflow JSON.

Persists to {data_dir}/workflow.json across restarts. Parses the workflow to
extract what it needs at runtime: LLM providers, files, embeddings, user_input.

Supports both camelCase (old Dare export) and snake_case (new export) field names.
"""
import json
import logging
from pathlib import Path

from core.loader import normalize_export

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
        # Normalise to the canonical shape so v1 and v2 parse identically.
        # An unsupported version shouldn't crash the dashboard — fall back to raw.
        try:
            payload = normalize_export(payload)
        except ValueError as e:
            logger.warning("Could not normalise workflow for info: %s", e)

        nodes = payload.get("nodes", [])

        required_providers = set()
        slots              = {}   # id -> {id, label, usage}; dedupes shared inputs
        needs_files        = False
        needs_embeddings   = False
        has_user_input     = False
        steps              = []

        def add_slot(f, default_usage):
            """Register a file slot keyed by its stable id (node id for v2,
            filename for v1). label is what the user sees; usage drives embedding."""
            fid = f.get("name") or str(f.get("fileId") or f.get("file_id", ""))
            if not fid:
                return
            slots.setdefault(fid, {
                "id":    fid,
                "label": f.get("label") or fid,
                "usage": f.get("usage") or default_usage,
            })

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
                text_input     = data.get("textInput") or ""
                if "{{user_input}}" in prompt_content or "{{user_input}}" in text_input:
                    has_user_input = True

                # Files attached directly to this step (canonical rag buckets)
                rag = data.get("rag", {})
                for f in rag.get("contentFiles", []):
                    needs_files = True
                    add_slot(f, "retrieve")
                for f in rag.get("embeddingFiles", []):
                    needs_files      = True
                    needs_embeddings = True
                    add_slot(f, "embed_and_retrieve")

                if ntype == "step":
                    steps.append({
                        "id":       node.get("id"),
                        "label":    data.get("label", "Step"),
                        "prompt":   prompt_content,
                        "provider": ks_provider,
                        "model":    llm.get("identifier", ""),
                    })

            elif ntype == "file":
                mode = data.get("retrievalMode", "content")
                embeds = mode in ("embeddings", "both")
                for f in data.get("files", []):
                    needs_files = True
                    add_slot(f, "embed_and_retrieve" if embeds else "retrieve")
                if embeds:
                    needs_embeddings = True

        return {
            "title":              payload.get("title", "Untitled Workflow"),
            "description":        payload.get("description", ""),
            "mode":               payload.get("mode", "sequential"),
            "steps":              steps,
            "required_providers": list(required_providers),
            "required_files":     list(slots.values()),
            "needs_files":        needs_files,
            "needs_embeddings":   needs_embeddings,
            "has_user_input":     has_user_input,
            "node_count":         len(nodes),
        }
