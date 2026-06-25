"""Config, file listing, and file-slot endpoints."""
import logging
from pathlib import Path

from fastapi import UploadFile, File as FastAPIFile, Form

from deps import (
    key_store, load_file_map, save_file_map, UPLOADS_DIR,
    embed_slot, remove_slot_embeddings, read_index_status, workflow_store,
)
from file_store import FileStore

logger = logging.getLogger(__name__)


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
    async def upload_slot_file(slot: str = Form(...), file: UploadFile = FastAPIFile(...)):
        """Upload a file to fill a workflow file slot.

        If the slot is an embedding input, the file is chunked + embedded into
        ChromaDB right away (keyed by the slot id) so retrieval works at run time.
        Embedding failures (e.g. no embed key yet) don't fail the upload — they're
        reported so the user can fix the key and re-upload.
        """
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        dest = UPLOADS_DIR / file.filename
        dest.write_bytes(await file.read())
        fmap = load_file_map()
        fmap[slot] = str(dest)
        save_file_map(fmap)

        resp = {"ok": True, "slot": slot, "file": file.filename, "embedded": False}
        if _slot_needs_embedding(slot):
            try:
                resp["chunks"] = embed_slot(slot)
                resp["embedded"] = True
            except Exception as e:
                logger.warning("Auto-embed failed for slot %s: %s", slot, e)
                resp["embed_error"] = str(e)
        return resp

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
        """Remove a slot mapping and its embeddings (the uploaded file stays on disk)."""
        fmap = load_file_map()
        fmap.pop(slot_name, None)
        save_file_map(fmap)
        remove_slot_embeddings(slot_name)
        return {"ok": True}
