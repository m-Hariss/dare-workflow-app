"""Workflow endpoints: upload / load-from-path / info / clear."""
import json
from pathlib import Path

from fastapi import HTTPException
from pydantic import BaseModel

from app_state import workflow_store, save_file_map
from core.loader import normalize_export


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
        workflow_store.save(payload)
        save_file_map({})
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
        workflow_store.save(payload)
        save_file_map({})
        return {"ok": True, "title": payload.get("title", "Untitled")}

    @app.get("/workflow/info", tags=["syftbox"])
    def workflow_info():
        info = workflow_store.get_info()
        if info is None:
            return {}
        return info

    @app.delete("/workflow/clear", tags=["syftbox"])
    def workflow_clear():
        workflow_store.clear()
        return {"ok": True}
