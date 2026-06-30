"""Workflow execution endpoint.

Runs are asynchronous: POST /run starts the workflow on a background thread and
returns a run_id immediately; the client polls GET /run/{run_id} for progress and
the final result. This keeps every HTTP request short, so it survives the SyftBox
embedded webview / RPC transport, which drops long-running requests.
"""
import copy
import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel
from syft_core import Client

from app_state import app_name, key_store, workflow_store, DATA_DIR, load_file_map, make_services
from core.engine import ExecutionEngine
from core.loader import load_workflow

logger = logging.getLogger(__name__)

# run_id -> {status, result, error, engine, total}. In-memory; single-process app.
_RUNS: dict = {}
_MAX_RUNS = 20  # keep the registry from growing unbounded


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


def _execute_run(run_id: str, payload: dict):
    """Background worker: build the graph, run the engine, store the outcome."""
    state = _RUNS[run_id]
    try:
        graph = load_workflow(payload)
    except ValueError as e:
        state.update(status="failed", error=f"Invalid workflow: {e}")
        return

    state["total"] = len(graph.nodes)
    try:
        services = make_services()
        engine = ExecutionEngine(graph, services)
        state["engine"] = engine          # exposes live per-node progress to the status endpoint
        result = engine.run()
        result["saved_to"] = _save_result(result)
        state.update(status=result.get("status", "completed"), result=result)
    except Exception as e:
        logger.exception("Run %s failed", run_id)
        state.update(status="failed", error=str(e))


def _prune_runs():
    while len(_RUNS) > _MAX_RUNS:
        _RUNS.pop(next(iter(_RUNS)))   # drop oldest (dicts keep insertion order)


def register(app):
    @app.post("/run", tags=["syftbox"])
    def run(data: RunRequest):
        payload = workflow_store.load()
        if payload is None:
            return {"error": "No workflow loaded. Upload a workflow JSON first."}

        if data.user_input:
            payload = _inject_user_input(payload, data.user_input)

        run_id = uuid.uuid4().hex[:12]
        _RUNS[run_id] = {"status": "running", "result": None, "error": None,
                         "engine": None, "total": 0}
        _prune_runs()
        threading.Thread(target=_execute_run, args=(run_id, payload), daemon=True).start()
        return {"run_id": run_id, "status": "running"}

    @app.get("/run/{run_id}", tags=["syftbox"])
    def run_status(run_id: str):
        state = _RUNS.get(run_id)
        if state is None:
            return {"error": "Unknown run id."}

        resp = {"status": state["status"]}

        engine = state.get("engine")
        if engine is not None:
            done = sum(1 for r in engine.state.results.values()
                       if r["status"] in ("completed", "failed", "skipped"))
            resp["progress"] = {"done": done, "total": state.get("total", 0)}

        if state["status"] == "failed" and not state.get("result"):
            resp["error"] = state.get("error") or "Run failed."
        if state["status"] != "running" and state.get("result") is not None:
            resp["result"] = state["result"]
        return resp
