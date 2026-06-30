"""Workflow endpoints: upload / load-from-path / info / clear."""
import json
import logging
from pathlib import Path

from fastapi import HTTPException
from pydantic import BaseModel

from app_state import workflow_store, save_file_map, load_file_map, remove_slot_embeddings, UPLOADS_DIR
from core.loader import normalize_export

logger = logging.getLogger(__name__)


def _wipe_slot_data():
    """Delete all uploaded files and embeddings for every current slot."""
    fmap = load_file_map()
    for slot, raw in fmap.items():
        paths = raw if isinstance(raw, list) else [raw]
        remove_slot_embeddings(slot)
        for path_str in paths:
            p = Path(path_str)
            try:
                if p.parent == UPLOADS_DIR and p.exists():
                    p.unlink()
                    logger.info("Deleted upload: %s", p)
            except Exception as e:
                logger.warning("Could not delete %s: %s", path_str, e)
    save_file_map({})
    logger.info("Wiped %d slot(s) on workflow clear", len(fmap))


class WorkflowPathRequest(BaseModel):
    path: str


class WorkflowUploadRequest(BaseModel):
    content: str   # raw JSON text of the exported workflow


def register(app):
    @app.post("/workflow/upload", tags=["syftbox"])
    def workflow_upload(body: WorkflowUploadRequest):
        try:
            payload = json.loads(body.content)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
        try:
            payload = normalize_export(payload)   # validate + normalize once, store normalized form
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        _wipe_slot_data()
        workflow_store.save(payload)
        return {"ok": True, "title": payload.get("title", "Untitled")}

    @app.post("/workflow/path", tags=["syftbox"])
    def workflow_from_path(body: WorkflowPathRequest):
        p = Path(body.path)
        if not p.exists():
            raise HTTPException(status_code=400, detail=f"File not found: {body.path}")
        try:
            payload = json.loads(p.read_text())
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
        try:
            payload = normalize_export(payload)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        _wipe_slot_data()
        workflow_store.save(payload)
        return {"ok": True, "title": payload.get("title", "Untitled")}

    @app.get("/workflow/info", tags=["syftbox"])
    def workflow_info():
        info = workflow_store.get_info()
        if info is None:
            return {}
        return info

    @app.delete("/workflow/clear", tags=["syftbox"])
    def workflow_clear():
        _wipe_slot_data()
        workflow_store.clear()
        return {"ok": True}
