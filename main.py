"""SyftBox app entry point.

Exposes:
  GET  /          — HTML dashboard (credentials UI + run status)
  GET  /ping      — health check
  GET  /keys/status — which providers are configured (no key values)
  POST /keys      — save one API key
  DELETE /keys/{provider} — remove a key
  POST /run       — execute an exported Dare workflow
"""
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import HTTPException
from fastapi.responses import HTMLResponse
from fastsyftbox import FastSyftBox
from pydantic import BaseModel
from syft_core import Client

from execution_engine import ExecutionEngine
from key_store import KeyStore
from services import Services
from workflow_loader import load_workflow

logger = logging.getLogger(__name__)

load_dotenv(Path(__file__).resolve().parent / ".env")

app_name = Path(__file__).resolve().parent.name

app = FastSyftBox(
    app_name=app_name,
    syftbox_endpoint_tags=["syftbox"],
    include_syft_openapi=True,
)


# ---------------------------------------------------------------------------
# KeyStore — initialised once at startup
# ---------------------------------------------------------------------------

def _init_key_store() -> KeyStore:
    try:
        client = Client.load()
        data_dir = Path(client.app_data(app_name))
    except Exception:
        data_dir = Path(__file__).resolve().parent / ".keystore"
    return KeyStore(data_dir)


key_store = _init_key_store()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class SaveKeyRequest(BaseModel):
    provider: str   # "openai" | "claude" | "gemini" | "ollama"
    key: str


class RunRequest(BaseModel):
    workflow: Optional[dict] = None
    workflowId: Optional[int] = None
    dareBaseUrl: Optional[str] = None
    dareToken: Optional[str] = None


# ---------------------------------------------------------------------------
# Key management endpoints
# ---------------------------------------------------------------------------

@app.get("/keys/status", tags=["syftbox"])
def keys_status():
    return key_store.status()


@app.post("/keys", tags=["syftbox"])
def save_key(body: SaveKeyRequest):
    if not body.provider or not body.key:
        raise HTTPException(status_code=400, detail="provider and key are required")
    key_store.set(body.provider.lower(), body.key.strip())
    return {"ok": True, "provider": body.provider.lower()}


@app.delete("/keys/{provider}", tags=["syftbox"])
def delete_key(provider: str):
    key_store.delete(provider.lower())
    return {"ok": True, "provider": provider.lower()}


# ---------------------------------------------------------------------------
# Run endpoint
# ---------------------------------------------------------------------------

def _save_result(result: dict) -> Optional[str]:
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
    services = Services.from_config(
        base_url=data.dareBaseUrl,
        token=data.dareToken,
        key_store=key_store,
    )

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


# ---------------------------------------------------------------------------
# HTML dashboard
# ---------------------------------------------------------------------------

_PROVIDER_META = {
    "openai":  {"label": "OpenAI",            "icon": "⚡", "placeholder": "sk-..."},
    "claude":  {"label": "Anthropic (Claude)", "icon": "🧠", "placeholder": "sk-ant-..."},
    "gemini":  {"label": "Google Gemini",      "icon": "✨", "placeholder": "AIza..."},
    "ollama":  {"label": "Ollama Host",        "icon": "🦙", "placeholder": "http://localhost:11434"},
}

_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dare Workflow Runner</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #f5f5f7;
    color: #1d1d1f;
    min-height: 100vh;
  }
  header {
    background: #1d1d1f;
    color: #f5f5f7;
    padding: 20px 32px;
    display: flex;
    align-items: center;
    gap: 12px;
  }
  header h1 { font-size: 1.2rem; font-weight: 600; letter-spacing: -0.2px; }
  header span { font-size: 0.8rem; opacity: 0.5; margin-left: 4px; }
  .ping-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #30d158; margin-left: auto;
    box-shadow: 0 0 0 2px rgba(48,209,88,.3);
  }
  main { max-width: 640px; margin: 40px auto; padding: 0 16px; }
  h2 { font-size: 1rem; font-weight: 600; color: #6e6e73; text-transform: uppercase;
       letter-spacing: .5px; margin-bottom: 16px; }
  .card {
    background: #fff;
    border-radius: 12px;
    margin-bottom: 12px;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,.08);
  }
  .card-header {
    display: flex; align-items: center; gap: 12px;
    padding: 16px 20px;
  }
  .icon { font-size: 1.4rem; width: 36px; text-align: center; flex-shrink: 0; }
  .provider-info { flex: 1; }
  .provider-info h3 { font-size: .95rem; font-weight: 600; margin-bottom: 2px; }
  .badge {
    display: inline-block; font-size: .75rem; font-weight: 500;
    padding: 2px 8px; border-radius: 20px;
  }
  .badge.ok   { background: #d1f9e0; color: #1a7f3c; }
  .badge.env  { background: #fef3c7; color: #92400e; }
  .badge.miss { background: #f1f1f4; color: #6e6e73; }
  .btn {
    border: none; border-radius: 8px; cursor: pointer;
    font-size: .85rem; font-weight: 500; padding: 7px 14px;
    transition: background .15s;
  }
  .btn-edit   { background: #f1f1f4; color: #1d1d1f; }
  .btn-edit:hover { background: #e3e3e8; }
  .btn-save   { background: #0071e3; color: #fff; }
  .btn-save:hover { background: #0077ed; }
  .btn-del    { background: #fff0f0; color: #d12f2f; }
  .btn-del:hover { background: #fde8e8; }
  .edit-row {
    display: none; padding: 0 20px 16px;
    gap: 8px; align-items: center; flex-wrap: wrap;
  }
  .edit-row.open { display: flex; }
  .edit-row input {
    flex: 1; min-width: 0;
    border: 1.5px solid #d2d2d7; border-radius: 8px;
    padding: 8px 12px; font-size: .9rem; outline: none;
    font-family: "SF Mono", ui-monospace, monospace;
    background: #f9f9fb;
  }
  .edit-row input:focus { border-color: #0071e3; background: #fff; }
  .msg { font-size: .8rem; padding: 6px 0; color: #6e6e73; flex-basis: 100%; }
  .msg.err { color: #d12f2f; }
  .msg.ok  { color: #1a7f3c; }
  footer { text-align: center; padding: 32px 0; font-size: .8rem; color: #aeaeb2; }
</style>
</head>
<body>
<header>
  <div>
    <h1>Dare Workflow Runner <span>SyftBox</span></h1>
  </div>
  <div class="ping-dot" id="pingDot" title="Checking…"></div>
</header>
<main>
  <h2>LLM API Keys</h2>
  <div id="cards">Loading…</div>
</main>
<footer>Keys are stored encrypted on this machine and never leave your device.</footer>

<script>
const PROVIDERS = {
  openai:  { label: "OpenAI",            icon: "⚡", ph: "sk-..." },
  claude:  { label: "Anthropic (Claude)", icon: "🧠", ph: "sk-ant-..." },
  gemini:  { label: "Google Gemini",      icon: "✨", ph: "AIza..." },
  ollama:  { label: "Ollama Host",        icon: "🦙", ph: "http://localhost:11434" },
};

async function loadStatus() {
  const r = await fetch("/keys/status");
  return r.json();
}

function badgeHtml(info) {
  if (!info.configured) return '<span class="badge miss">Not set</span>';
  if (info.source === "env")   return '<span class="badge env">Set via env var</span>';
  return '<span class="badge ok">Configured ✓</span>';
}

function renderCards(status) {
  const container = document.getElementById("cards");
  container.innerHTML = "";
  for (const [id, meta] of Object.entries(PROVIDERS)) {
    const info = status[id] || { configured: false, source: null };
    const card = document.createElement("div");
    card.className = "card";
    card.innerHTML = `
      <div class="card-header">
        <div class="icon">${meta.icon}</div>
        <div class="provider-info">
          <h3>${meta.label}</h3>
          ${badgeHtml(info)}
        </div>
        <button class="btn btn-edit" onclick="toggleEdit('${id}')">
          ${info.configured ? "Update" : "Add"}
        </button>
      </div>
      <div class="edit-row" id="edit-${id}">
        <input type="password" id="input-${id}" placeholder="${meta.ph}" autocomplete="off" />
        <button class="btn btn-save" onclick="saveKey('${id}')">Save</button>
        ${info.configured ? `<button class="btn btn-del" onclick="deleteKey('${id}')">Remove</button>` : ""}
        <div class="msg" id="msg-${id}"></div>
      </div>`;
    container.appendChild(card);
  }
}

function toggleEdit(id) {
  const row = document.getElementById("edit-" + id);
  const isOpen = row.classList.contains("open");
  // Close all
  document.querySelectorAll(".edit-row.open").forEach(r => r.classList.remove("open"));
  if (!isOpen) {
    row.classList.add("open");
    document.getElementById("input-" + id).focus();
  }
}

function setMsg(id, text, isErr) {
  const el = document.getElementById("msg-" + id);
  el.textContent = text;
  el.className = "msg " + (isErr ? "err" : "ok");
  if (!isErr) setTimeout(() => { el.textContent = ""; refresh(); }, 1400);
}

async function saveKey(provider) {
  const key = document.getElementById("input-" + provider).value.trim();
  if (!key) { setMsg(provider, "Enter a value first.", true); return; }
  const r = await fetch("/keys", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ provider, key }),
  });
  if (r.ok) setMsg(provider, "Saved!", false);
  else setMsg(provider, "Error saving key.", true);
}

async function deleteKey(provider) {
  const r = await fetch("/keys/" + provider, { method: "DELETE" });
  if (r.ok) { setMsg(provider, "Removed.", false); }
  else setMsg(provider, "Error removing key.", true);
}

async function refresh() {
  const status = await loadStatus();
  renderCards(status);
}

// Ping dot
async function checkPing() {
  try {
    const r = await fetch("/ping");
    document.getElementById("pingDot").style.background = r.ok ? "#30d158" : "#ff453a";
  } catch { document.getElementById("pingDot").style.background = "#ff453a"; }
}

checkPing();
refresh();
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse, tags=["syftbox"])
def root():
    return HTMLResponse(_HTML)


@app.get("/ping", tags=["syftbox"])
def ping():
    return {"status": "ok", "app": app_name}
