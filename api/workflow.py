"""Workflow endpoints: upload / load-from-path / info / clear.

Registered via decorators on the app (not an APIRouter): FastSyftBox discovers
syft-published endpoints by scanning top-level app.routes for APIRoute instances,
and the installed FastAPI nests include_router() routes out of that view.
"""
import json
from pathlib import Path

from fastapi import HTTPException, UploadFile, File as FastAPIFile
from pydantic import BaseModel

from deps import workflow_store, save_file_map


class WorkflowPathRequest(BaseModel):
    path: str


def register(app):
    @app.post("/workflow/upload", tags=["syftbox"])
    async def workflow_upload(file: UploadFile = FastAPIFile(...)):
        content = await file.read()
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
        workflow_store.save(payload)
        save_file_map({})  # clear slot mappings from any previous workflow
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
        workflow_store.save(payload)
        save_file_map({})  # clear slot mappings from any previous workflow
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
