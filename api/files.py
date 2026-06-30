"""Config, file listing, and file-slot endpoints."""
import asyncio
import base64
import logging
from pathlib import Path

from fastapi import HTTPException
from pydantic import BaseModel

from app_state import (
    key_store, load_file_map, save_file_map, UPLOADS_DIR,
    embed_slot, remove_slot_embeddings, read_index_status, workflow_store,
    get_embed_progress,
)
from storage.file_store import FileStore

logger = logging.getLogger(__name__)


class SlotUploadRequest(BaseModel):
    slot: str
    filename: str
    content_b64: str


class SlotEmbedRequest(BaseModel):
    slot: str


def _slot_needs_embedding(slot_id: str) -> bool:
    info = workflow_store.get_info() or {}
    for s in info.get("required_files", []):
        if s.get("id") == slot_id:
            return "embed" in (s.get("usage") or "")
    return False


def _slot_paths(raw) -> list[str]:
    """Normalise a file_map value to a list of path strings."""
    if raw is None:
        return []
    return raw if isinstance(raw, list) else [raw]


def register(app):
    @app.get("/config", tags=["syftbox"])
    def get_config():
        folder = key_store.get("files_folder")
        store  = FileStore(folder)
        return {
            "files_folder":       folder,
            "files_folder_valid": store.is_configured(),
        }

    @app.get("/files", tags=["syftbox"])
    def list_files():
        folder = key_store.get("files_folder")
        store  = FileStore(folder)
        return {
            "folder": folder,
            "valid":  store.is_configured(),
            "files":  store.list_files(),
        }

    # ── File slot endpoints ─────────────────────────────────────────────

    @app.post("/files/slot", tags=["syftbox"])
    def upload_slot_file(body: SlotUploadRequest):
        """Upload a file to a workflow slot. Multiple uploads append to the slot."""
        try:
            raw = base64.b64decode(body.content_b64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid file content: {e}")

        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        dest = UPLOADS_DIR / Path(body.filename).name
        dest.write_bytes(raw)

        fmap = load_file_map()
        existing = _slot_paths(fmap.get(body.slot))
        # Avoid duplicates — replace if same filename already in slot
        existing = [p for p in existing if Path(p).name != Path(body.filename).name]
        fmap[body.slot] = existing + [str(dest)]
        save_file_map(fmap)

        return {
            "ok":              True,
            "slot":            body.slot,
            "file":            dest.name,
            "needs_embedding": _slot_needs_embedding(body.slot),
            "total_files":     len(fmap[body.slot]),
        }

    @app.post("/files/slot/embed", tags=["syftbox"])
    async def embed_slot_file(body: SlotEmbedRequest):
        """Chunk + embed all files in a slot into ChromaDB."""
        logger.info("[/files/slot/embed] request received  slot=%s", body.slot)
        def _run():
            try:
                chunks = embed_slot(body.slot)
                return {"ok": True, "embedded": True, "chunks": chunks}
            except Exception as e:
                logger.warning("[/files/slot/embed] failed  slot=%s  error=%s", body.slot, e)
                return {"ok": True, "embedded": False, "embed_error": str(e)}
        result = await asyncio.to_thread(_run)
        logger.info("[/files/slot/embed] done  slot=%s  result=%s", body.slot, result)
        return result

    @app.get("/files/slot/embed/progress/{slot_name:path}", tags=["syftbox"])
    def embed_progress(slot_name: str):
        """Return real-time embedding progress for a slot."""
        return get_embed_progress(slot_name)

    @app.get("/files/slots", tags=["syftbox"])
    def get_file_slots():
        """Return current slot → uploaded file mappings with index status."""
        fmap   = load_file_map()
        status = read_index_status()
        out = {}
        for slot, raw in fmap.items():
            paths = _slot_paths(raw)
            files = [
                {"path": p, "filename": Path(p).name, "exists": Path(p).exists()}
                for p in paths
            ]
            entry = {
                "files":  files,
                "count":  len(files),
                "exists": any(f["exists"] for f in files),
            }
            if slot in status:
                entry["indexed"] = True
                entry["chunks"]  = status[slot].get("chunks", 0)
            out[slot] = entry
        return out

    @app.delete("/files/slot/{slot_name:path}", tags=["syftbox"])
    def delete_file_slot(slot_name: str):
        """Remove a slot: its mapping, embeddings, and uploaded files on disk."""
        fmap = load_file_map()
        raw  = fmap.pop(slot_name, None)
        save_file_map(fmap)
        remove_slot_embeddings(slot_name)

        if raw:
            remaining_paths = set(p for v in fmap.values() for p in _slot_paths(v))
            for path_str in _slot_paths(raw):
                p = Path(path_str)
                try:
                    if p.parent == UPLOADS_DIR and p.exists() and path_str not in remaining_paths:
                        p.unlink()
                except Exception as e:
                    logger.warning("Could not delete uploaded file %s: %s", path_str, e)
        return {"ok": True}
