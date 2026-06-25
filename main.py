"""SyftBox app — Dare Workflow Runner.

Thin entry point: builds the FastSyftBox app, mounts the API routers, and serves
the dashboard + health check. Route handlers live in api/, shared state in deps.py.

Endpoints:
  GET  /                       HTML dashboard
  GET  /ping                   health check

  POST   /workflow/upload      upload a workflow JSON file
  POST   /workflow/path        load workflow from a local file path
  GET    /workflow/info        parsed workflow metadata
  DELETE /workflow/clear       remove stored workflow

  GET    /keys/status          configured LLM providers
  POST   /keys                 save a key / config value
  DELETE /keys/{provider}      remove a key

  GET  /config                 non-secret config (files folder, embed model)
  GET  /files                  list files in the configured folder
  POST   /files/slot           upload a file into a workflow file slot
  GET    /files/slots          current slot -> file mappings
  DELETE /files/slot/{slot}    remove a slot mapping

  POST /index                  chunk + embed all files into ChromaDB
  GET  /index/status           per-file index state

  POST /run                    run the stored workflow
"""
import logging
from pathlib import Path

from fastapi.responses import HTMLResponse
from fastsyftbox import FastSyftBox

from deps import app_name
from api import files, index, keys, run, workflow

logger = logging.getLogger(__name__)

app = FastSyftBox(
    app_name=app_name,
    syftbox_endpoint_tags=["syftbox"],
    include_syft_openapi=True,
)

workflow.register(app)
keys.register(app)
files.register(app)
index.register(app)
run.register(app)

_DASHBOARD = (Path(__file__).resolve().parent / "web" / "dashboard.html").read_text()


@app.get("/", response_class=HTMLResponse, tags=["syftbox"])
def root():
    return HTMLResponse(_DASHBOARD)


@app.get("/ping", tags=["syftbox"])
def ping():
    return {"status": "ok", "app": app_name}
