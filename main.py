"""SyftBox app entry point.

Exposes a /run endpoint that executes an exported Dare workflow. The workflow can
be passed inline or pulled from Dare by id. All file/retrieval access is delegated
back to Dare; the runner performs no embedding/indexing/storage itself.
"""
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi.responses import HTMLResponse
from fastsyftbox import FastSyftBox
from pydantic import BaseModel
from syft_core import Client

from execution_engine import ExecutionEngine
from services import Services
from workflow_loader import load_workflow

logger = logging.getLogger(__name__)

# Load credentials (DARE_TOKEN, OPENAI_API_KEY, ...) from a local .env so the
# SyftBox-managed process has them even though it isn't launched from a shell
# with those vars exported.
load_dotenv(Path(__file__).resolve().parent / ".env")

app_name = Path(__file__).resolve().parent.name

app = FastSyftBox(
    app_name=app_name,
    syftbox_endpoint_tags=["syftbox"],
    include_syft_openapi=True,
)


class RunRequest(BaseModel):
    workflow: Optional[dict] = None      # inline exported graph
    workflowId: Optional[int] = None     # or fetch from Dare by id
    dareBaseUrl: Optional[str] = None
    dareToken: Optional[str] = None


@app.get("/", response_class=HTMLResponse)
def root():
    return HTMLResponse(f"<html><body><h1>{app_name}</h1>{app.get_debug_urls()}</body></html>")


@app.get("/ping", tags=["syftbox"])
def ping():
    return {"status": "ok", "app": app_name}


def _save_result(result: dict) -> Optional[str]:
    """Persist a run result into the app's datasite so it can be viewed on SyftBox.

    Returns the file path, or None if saving failed (never blocks the run).
    """
    try:
        client = Client.load()
        runs_dir = Path(client.app_data(app_name)) / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        wid = result.get("workflow_id", "unknown")
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        path = runs_dir / f"run-wf{wid}-{ts}-{uuid.uuid4().hex[:6]}.json"
        path.write_text(json.dumps(result, indent=2, default=str))
        return str(path)
    except Exception as e:
        logger.warning("Could not save run result to datasite: %s", e)
        return None


@app.api_route("/run", methods=["POST"], tags=["syftbox"])
def run(data: RunRequest):
    services = Services.from_config(base_url=data.dareBaseUrl, token=data.dareToken)

    if data.workflow is not None:
        payload = data.workflow
    elif data.workflowId is not None:
        payload = services.dare.export_workflow(data.workflowId)
    else:
        return {"error": "Provide 'workflow' (inline export) or 'workflowId'."}

    try:
        graph = load_workflow(payload)
    except ValueError as e:
        return {"error": f"Invalid workflow export: {e}"}

    result = ExecutionEngine(graph, services).run()
    result["saved_to"] = _save_result(result)
    return result
