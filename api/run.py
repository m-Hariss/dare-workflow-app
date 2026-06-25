"""Workflow execution endpoint."""
import copy
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel
from syft_core import Client

from deps import app_name, key_store, workflow_store, DATA_DIR, load_file_map
from execution_engine import ExecutionEngine
from services import Services
from workflow_loader import load_workflow

logger = logging.getLogger(__name__)


class RunRequest(BaseModel):
    user_input: Optional[str] = None


def _inject_user_input(workflow: dict, user_input: str) -> dict:
    """Replace {{user_input}} in all prompt content and text input fields."""
    workflow = copy.deepcopy(workflow)
    for node in workflow.get("nodes", []):
        data = node.get("data", {})
        prompt = data.get("prompt") or {}
        if isinstance(prompt.get("content"), str):
            prompt["content"] = prompt["content"].replace("{{user_input}}", user_input)
        for key in ("textInput", "text_input"):
            if isinstance(data.get(key), str):
                data[key] = data[key].replace("{{user_input}}", user_input)
    return workflow


def _save_result(result: dict) -> Optional[str]:
    try:
        client   = Client.load()
        runs_dir = Path(client.app_data(app_name)) / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        wid  = result.get("workflow_id", "unknown")
        ts   = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        path = runs_dir / f"run-wf{wid}-{ts}-{uuid.uuid4().hex[:6]}.json"
        path.write_text(json.dumps(result, indent=2, default=str))
        return str(path)
    except Exception as e:
        logger.warning("Could not save run result: %s", e)
        return None


def register(app):
    @app.post("/run", tags=["syftbox"])
    def run(data: RunRequest):
        payload = workflow_store.load()
        if payload is None:
            return {"error": "No workflow loaded. Upload a workflow JSON first."}

        if data.user_input:
            payload = _inject_user_input(payload, data.user_input)

        try:
            graph = load_workflow(payload)
        except ValueError as e:
            return {"error": f"Invalid workflow: {e}"}

        services = Services.from_config(key_store=key_store, data_dir=DATA_DIR, file_map=load_file_map())
        result   = ExecutionEngine(graph, services).run()
        result["saved_to"] = _save_result(result)
        return result
