"""Config, file listing, and file-slot endpoints."""
from pathlib import Path

from fastapi import UploadFile, File as FastAPIFile, Form

from deps import key_store, load_file_map, save_file_map, UPLOADS_DIR
from file_store import FileStore


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
        """Upload a file to fill a workflow file slot."""
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        dest = UPLOADS_DIR / file.filename
        dest.write_bytes(await file.read())
        fmap = load_file_map()
        fmap[slot] = str(dest)
        save_file_map(fmap)
        return {"ok": True, "slot": slot, "file": file.filename}

    @app.get("/files/slots", tags=["syftbox"])
    def get_file_slots():
        """Return current slot → uploaded file mappings."""
        fmap = load_file_map()
        return {
            slot: {"path": path, "filename": Path(path).name, "exists": Path(path).exists()}
            for slot, path in fmap.items()
        }

    @app.delete("/files/slot/{slot_name:path}", tags=["syftbox"])
    def delete_file_slot(slot_name: str):
        """Remove a slot mapping (does not delete the uploaded file from disk)."""
        fmap = load_file_map()
        fmap.pop(slot_name, None)
        save_file_map(fmap)
        return {"ok": True}
