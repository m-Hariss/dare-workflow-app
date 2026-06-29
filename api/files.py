"""Config, file listing, and file-slot endpoints."""
import base64
import logging
from pathlib import Path

from fastapi import HTTPException
from pydantic import BaseModel

from deps import (
    key_store, load_file_map, save_file_map, UPLOADS_DIR,
    embed_slot, remove_slot_embeddings, read_index_status, workflow_store,
)
from storage.file_store import FileStore

logger = logging.getLogger(__name__)


class SlotUploadRequest(BaseModel):
    slot: str
    filename: str
    content_b64: str   # base64 of the file bytes — JSON-safe so it rides the syft:// RPC transport


class SlotEmbedRequest(BaseModel):
    slot: str


def _slot_needs_embedding(slot_id: str) -> bool:
    """True if the active workflow declares this slot as an embedding input."""
    info = workflow_store.get_info() or {}
    for s in info.get("required_files", []):
        if s.get("id") == slot_id:
            return "embed" in (s.get("usage") or "")
    return False


def register(app):
    @app.get("/config", tags=["syftbox"])
    def get_config():
        folder = key_store.get("files_folder")
        store  = FileStore(folder)
        return {
            "files_folder":       folder,
            "files_folder_valid": store.is_configured(),
            "embed_provider":     key_store.get("embed_provider") or "openai",
            "embed_model":        key_store.get("embed_model")    or "text-embedding-3-small",
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

    # File slot endpoints — let users upload their own files to replace workflow references

    @app.post("/files/slot", tags=["syftbox"])
    def upload_slot_file(body: SlotUploadRequest):
        """Upload a file (base64-in-JSON) to fill a workflow file slot.

        JSON rather than multipart so the upload works over the SyftBox syft://
        RPC transport (which carries JSON, not multipart binary), not just plain HTTP.

        This only saves the file. Embedding is a separate call (/files/slot/embed)
        so the UI can show "uploading" then "generating embeddings" as distinct
        phases. `needs_embedding` tells the client whether to follow up.
        """
        try:
            raw = base64.b64decode(body.content_b64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid file content: {e}")

        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        dest = UPLOADS_DIR / Path(body.filename).name
        dest.write_bytes(raw)
        fmap = load_file_map()
        fmap[body.slot] = str(dest)
        save_file_map(fmap)

        return {
            "ok": True,
            "slot": body.slot,
            "file": dest.name,
            "needs_embedding": _slot_needs_embedding(body.slot),
        }

    @app.post("/files/slot/embed", tags=["syftbox"])
    def embed_slot_file(body: SlotEmbedRequest):
        """Chunk + embed an already-uploaded slot file into ChromaDB.

        Called by the UI right after upload (for embed slots). Kept separate so the
        upload returns immediately and the embedding runs as its own visible phase.
        Failures (e.g. no embed key) are reported, not raised — the file stays.
        """
        try:
            chunks = embed_slot(body.slot)
            return {"ok": True, "embedded": True, "chunks": chunks}
        except Exception as e:
            logger.warning("Embed failed for slot %s: %s", body.slot, e)
            return {"ok": True, "embedded": False, "embed_error": str(e)}

    @app.get("/files/slots", tags=["syftbox"])
    def get_file_slots():
        """Return current slot → uploaded file mappings, with index status."""
        fmap   = load_file_map()
        status = read_index_status()
        out = {}
        for slot, path in fmap.items():
            entry = {
                "path":     path,
                "filename": Path(path).name,
                "exists":   Path(path).exists(),
            }
            if slot in status:
                entry["indexed"] = True
                entry["chunks"]  = status[slot].get("chunks", 0)
            out[slot] = entry
        return out

    @app.delete("/files/slot/{slot_name:path}", tags=["syftbox"])
    def delete_file_slot(slot_name: str):
        """Remove a slot: its mapping, its embeddings, and the uploaded file on disk.

        The physical file is only deleted if it lives in our uploads/ folder and no
        other slot still points at it (never touches files in a user's files folder).
        """
        fmap = load_file_map()
        path = fmap.pop(slot_name, None)
        save_file_map(fmap)
        remove_slot_embeddings(slot_name)

        if path:
            p = Path(path)
            try:
                if p.parent == UPLOADS_DIR and p.exists() and path not in fmap.values():
                    p.unlink()
            except Exception as e:
                logger.warning("Could not delete uploaded file %s: %s", path, e)
        return {"ok": True}
