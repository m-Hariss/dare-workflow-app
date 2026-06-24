"""SyftBox app entry point.

Exposes:
  GET  /                    — HTML dashboard
  GET  /ping                — health check
  GET  /keys/status         — which LLM providers are configured (no key values)
  POST /keys                — save an API key / config value
  DELETE /keys/{provider}   — remove a key
  GET  /config              — non-secret config (files_folder, embed_provider, embed_model)
  GET  /files               — list files in the configured folder
  POST /index               — chunk + embed + store all files in ChromaDB
  GET  /index/status        — which files are indexed, model used, timestamp
  POST /run                 — execute an exported Dare workflow
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

from embedding_pipeline import EmbeddingPipeline
from execution_engine import ExecutionEngine
from file_store import FileStore
from key_store import KeyStore
from services import Services
from vector_store import VectorStore
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
# Shared singletons (initialised once at startup)
# ---------------------------------------------------------------------------

def _init_data_dir() -> Path:
    try:
        client = Client.load()
        return Path(client.app_data(app_name))
    except Exception:
        return Path(__file__).resolve().parent / ".local_data"


_DATA_DIR = _init_data_dir()
key_store = KeyStore(_DATA_DIR)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class SaveKeyRequest(BaseModel):
    provider: str
    key: str


class RunRequest(BaseModel):
    workflow: Optional[dict] = None
    workflowId: Optional[int] = None
    dareBaseUrl: Optional[str] = None
    dareToken: Optional[str] = None


# ---------------------------------------------------------------------------
# Key management
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
# Config
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

@app.get("/files", tags=["syftbox"])
def list_files():
    folder = key_store.get("files_folder")
    store  = FileStore(folder)
    return {
        "folder": folder,
        "valid":  store.is_configured(),
        "files":  store.list_files(),
    }


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def _make_embed_pipeline() -> EmbeddingPipeline:
    provider = key_store.get("embed_provider") or "openai"
    model    = key_store.get("embed_model")    or "text-embedding-3-small"
    api_keys = {}
    for p in ("openai", "claude", "gemini"):
        v = key_store.get(p)
        if v:
            api_keys[p] = v
    ollama_host = key_store.get("ollama")
    from clients.llm_client import LLMClient
    llm = LLMClient(api_keys=api_keys or None, ollama_host=ollama_host)
    return EmbeddingPipeline(
        vector_store=VectorStore(_DATA_DIR),
        llm_client=llm,
        data_dir=_DATA_DIR,
        provider=provider,
        model=model,
    )


@app.post("/index", tags=["syftbox"])
def index_files():
    folder = key_store.get("files_folder")
    file_store = FileStore(folder)

    if not file_store.is_configured():
        return {"error": "No files folder configured. Set it in the dashboard first."}

    files = file_store.list_files()
    if not files:
        return {"error": "Files folder is empty — nothing to index."}

    pipeline = _make_embed_pipeline()
    indexed, failed = [], []

    for f in files:
        try:
            content = file_store.get_content(f["name"])
            n_chunks = pipeline.index_file(f["name"], content)
            indexed.append({"name": f["name"], "chunks": n_chunks})
            logger.info("Indexed %s (%d chunks)", f["name"], n_chunks)
        except Exception as e:
            logger.error("Failed to index %s: %s", f["name"], e)
            failed.append({"name": f["name"], "error": str(e)})

    return {
        "indexed": indexed,
        "failed":  failed,
        "total_chunks": sum(i["chunks"] for i in indexed),
    }


@app.get("/index/status", tags=["syftbox"])
def index_status():
    pipeline = _make_embed_pipeline()
    return pipeline.get_status()


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def _save_result(result: dict) -> Optional[str]:
    try:
        client  = Client.load()
        runs_dir = Path(client.app_data(app_name)) / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        wid  = result.get("workflow_id", "unknown")
        ts   = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
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
        data_dir=_DATA_DIR,
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
    background: #f5f5f7; color: #1d1d1f; min-height: 100vh;
  }
  header {
    background: #1d1d1f; color: #f5f5f7;
    padding: 20px 32px; display: flex; align-items: center; gap: 12px;
  }
  header h1 { font-size: 1.2rem; font-weight: 600; }
  header span { font-size: .8rem; opacity: .5; margin-left: 4px; }
  .ping-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #30d158; margin-left: auto;
    box-shadow: 0 0 0 2px rgba(48,209,88,.3);
  }
  main { max-width: 640px; margin: 40px auto; padding: 0 16px 60px; }
  h2 {
    font-size: .78rem; font-weight: 600; color: #6e6e73;
    text-transform: uppercase; letter-spacing: .5px; margin: 32px 0 10px;
  }
  h2:first-child { margin-top: 0; }
  .card {
    background: #fff; border-radius: 12px; margin-bottom: 10px;
    overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,.08);
  }
  .card-header {
    display: flex; align-items: center; gap: 12px; padding: 16px 20px;
  }
  .icon { font-size: 1.3rem; width: 34px; text-align: center; flex-shrink: 0; }
  .info { flex: 1; }
  .info h3 { font-size: .9rem; font-weight: 600; margin-bottom: 3px; }
  .info p  { font-size: .78rem; color: #6e6e73; word-break: break-all; }
  .badge {
    display: inline-block; font-size: .72rem; font-weight: 500;
    padding: 2px 8px; border-radius: 20px;
  }
  .badge.ok   { background: #d1f9e0; color: #1a7f3c; }
  .badge.env  { background: #fef3c7; color: #92400e; }
  .badge.miss { background: #f1f1f4; color: #6e6e73; }
  .btn {
    border: none; border-radius: 8px; cursor: pointer;
    font-size: .82rem; font-weight: 500; padding: 7px 14px;
    transition: background .15s; white-space: nowrap;
  }
  .btn:disabled { opacity: .5; cursor: not-allowed; }
  .btn-edit  { background: #f1f1f4; color: #1d1d1f; }
  .btn-edit:hover:not(:disabled) { background: #e3e3e8; }
  .btn-save  { background: #0071e3; color: #fff; }
  .btn-save:hover:not(:disabled) { background: #0077ed; }
  .btn-del   { background: #fff0f0; color: #d12f2f; }
  .btn-del:hover:not(:disabled)  { background: #fde8e8; }
  .btn-index { background: #5856d6; color: #fff; }
  .btn-index:hover:not(:disabled) { background: #4845c4; }
  .edit-row {
    display: none; padding: 0 20px 16px;
    gap: 8px; align-items: center; flex-wrap: wrap;
  }
  .edit-row.open { display: flex; }
  .edit-row input, .edit-row select {
    flex: 1; min-width: 0;
    border: 1.5px solid #d2d2d7; border-radius: 8px;
    padding: 8px 12px; font-size: .85rem; outline: none; background: #f9f9fb;
  }
  .edit-row input:focus, .edit-row select:focus {
    border-color: #0071e3; background: #fff;
  }
  .msg { font-size: .78rem; padding: 4px 0; color: #6e6e73; flex-basis: 100%; }
  .msg.err { color: #d12f2f; }
  .msg.ok  { color: #1a7f3c; }
  .file-list { padding: 0 20px 16px; }
  .file-empty { font-size: .82rem; color: #aeaeb2; padding: 8px 0; }
  .file-item {
    display: flex; align-items: center; gap: 8px;
    padding: 7px 0; border-bottom: 1px solid #f1f1f4; font-size: .85rem;
  }
  .file-item:last-child { border-bottom: none; }
  .file-name { flex: 1; font-family: "SF Mono", ui-monospace, monospace; font-size: .8rem; }
  .file-meta { color: #aeaeb2; font-size: .72rem; white-space: nowrap; }
  .index-bar {
    padding: 14px 20px; display: flex;
    justify-content: space-between; align-items: center; gap: 12px;
  }
  .index-bar span { font-size: .85rem; color: #6e6e73; }
  footer { text-align: center; padding: 32px 0; font-size: .78rem; color: #aeaeb2; }
</style>
</head>
<body>
<header>
  <div><h1>Dare Workflow Runner <span>SyftBox</span></h1></div>
  <div class="ping-dot" id="pingDot"></div>
</header>
<main>

  <!-- ── LLM Keys ──────────────────────────────────────── -->
  <h2>LLM API Keys</h2>
  <div id="keysSection">Loading…</div>

  <!-- ── Files Folder ─────────────────────────────────── -->
  <h2>Files Folder</h2>
  <div class="card">
    <div class="card-header">
      <div class="icon">📁</div>
      <div class="info">
        <h3>Source folder</h3>
        <p id="folderPath" style="color:#aeaeb2">Loading…</p>
      </div>
      <button class="btn btn-edit" onclick="toggleFolder()">Change</button>
    </div>
    <div class="edit-row" id="folderEditRow">
      <input type="text" id="folderInput" placeholder="/Users/you/Documents/my-project/" />
      <button class="btn btn-save" onclick="saveFolder()">Save</button>
      <div class="msg" id="folderMsg"></div>
    </div>
  </div>

  <h2>Files in Folder</h2>
  <div class="card">
    <div id="fileList" class="file-list">
      <p class="file-empty">Set a files folder above to see available files.</p>
    </div>
  </div>

  <!-- ── Embeddings ───────────────────────────────────── -->
  <h2>Embedding Model</h2>
  <div class="card">
    <div class="card-header">
      <div class="icon">🔍</div>
      <div class="info">
        <h3>Provider &amp; model</h3>
        <p id="embedInfo" style="color:#aeaeb2">Loading…</p>
      </div>
      <button class="btn btn-edit" onclick="toggleEmbed()">Configure</button>
    </div>
    <div class="edit-row" id="embedEditRow">
      <select id="embedProvider" onchange="updateEmbedPlaceholder()">
        <option value="openai">OpenAI</option>
        <option value="ollama">Ollama</option>
      </select>
      <input type="text" id="embedModel" placeholder="text-embedding-3-small" />
      <button class="btn btn-save" onclick="saveEmbed()">Save</button>
      <div class="msg" id="embedMsg"></div>
    </div>
  </div>

  <!-- ── Index Status ─────────────────────────────────── -->
  <h2>Index Status</h2>
  <div class="card">
    <div class="index-bar">
      <span id="indexSummary">Loading…</span>
      <button class="btn btn-index" id="indexBtn" onclick="indexFiles()">Index All Files</button>
    </div>
    <div id="indexList" class="file-list" style="padding-top:0"></div>
  </div>

</main>
<footer>API keys are encrypted on this machine and never leave your device.</footer>

<script>
/* ── LLM Keys ────────────────────────────────────────────────── */
const PROVIDERS = {
  openai: { label:"OpenAI",            icon:"⚡", ph:"sk-...",                  type:"password" },
  claude: { label:"Anthropic (Claude)", icon:"🧠", ph:"sk-ant-...",             type:"password" },
  gemini: { label:"Google Gemini",      icon:"✨", ph:"AIza...",               type:"password" },
  ollama: { label:"Ollama Host",        icon:"🦙", ph:"http://localhost:11434", type:"text"     },
};

function badgeHtml(info) {
  if (!info.configured)          return '<span class="badge miss">Not set</span>';
  if (info.source === "env")     return '<span class="badge env">Via env var</span>';
  return '<span class="badge ok">Configured ✓</span>';
}

function renderKeys(status) {
  const c = document.getElementById("keysSection");
  c.innerHTML = "";
  for (const [id, meta] of Object.entries(PROVIDERS)) {
    const info = status[id] || { configured:false, source:null };
    const card = document.createElement("div");
    card.className = "card";
    card.innerHTML = `
      <div class="card-header">
        <div class="icon">${meta.icon}</div>
        <div class="info"><h3>${meta.label}</h3>${badgeHtml(info)}</div>
        <button class="btn btn-edit" onclick="toggleKey('${id}')">${info.configured?"Update":"Add"}</button>
      </div>
      <div class="edit-row" id="edit-${id}">
        <input type="${meta.type}" id="input-${id}" placeholder="${meta.ph}" autocomplete="off"/>
        <button class="btn btn-save" onclick="saveKey('${id}')">Save</button>
        ${info.configured?`<button class="btn btn-del" onclick="deleteKey('${id}')">Remove</button>`:""}
        <div class="msg" id="msg-${id}"></div>
      </div>`;
    c.appendChild(card);
  }
}

function toggleKey(id) {
  const row = document.getElementById("edit-"+id);
  const open = row.classList.contains("open");
  document.querySelectorAll(".edit-row.open").forEach(r=>r.classList.remove("open"));
  if (!open) { row.classList.add("open"); document.getElementById("input-"+id).focus(); }
}

function setMsg(id, text, isErr) {
  const el = document.getElementById("msg-"+id);
  el.textContent = text; el.className = "msg "+(isErr?"err":"ok");
  if (!isErr) setTimeout(()=>{ el.textContent=""; refreshKeys(); }, 1400);
}

async function saveKey(p) {
  const key = document.getElementById("input-"+p).value.trim();
  if (!key) { setMsg(p,"Enter a value first.",true); return; }
  const r = await fetch("/keys",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({provider:p,key})});
  r.ok ? setMsg(p,"Saved!",false) : setMsg(p,"Error saving.",true);
}

async function deleteKey(p) {
  const r = await fetch("/keys/"+p,{method:"DELETE"});
  r.ok ? setMsg(p,"Removed.",false) : setMsg(p,"Error.",true);
}

async function refreshKeys() {
  const s = await fetch("/keys/status").then(r=>r.json());
  renderKeys(s);
}

/* ── Files Folder ───────────────────────────────────────────── */
async function refreshFolder() {
  const cfg = await fetch("/config").then(r=>r.json());
  const el = document.getElementById("folderPath");
  if (cfg.files_folder) {
    el.textContent = cfg.files_folder;
    el.style.color = cfg.files_folder_valid ? "#1d1d1f" : "#d12f2f";
    if (!cfg.files_folder_valid) el.textContent += " ⚠ path not found";
  } else {
    el.textContent = "Not configured";
    el.style.color = "#aeaeb2";
  }
  refreshFileList();
}

function toggleFolder() {
  const row = document.getElementById("folderEditRow");
  row.classList.contains("open") ? row.classList.remove("open") : (row.classList.add("open"), document.getElementById("folderInput").focus());
}

async function saveFolder() {
  const path = document.getElementById("folderInput").value.trim();
  if (!path) { folderMsg("Enter a folder path.",true); return; }
  const r = await fetch("/keys",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({provider:"files_folder",key:path})});
  if (r.ok) {
    folderMsg("Saved!",false);
    setTimeout(()=>{ document.getElementById("folderEditRow").classList.remove("open"); document.getElementById("folderMsg").textContent=""; refreshFolder(); }, 900);
  } else folderMsg("Error.",true);
}

function folderMsg(t,e){ const el=document.getElementById("folderMsg"); el.textContent=t; el.className="msg "+(e?"err":"ok"); }

async function refreshFileList() {
  const data = await fetch("/files").then(r=>r.json());
  const el = document.getElementById("fileList");
  if (!data.folder)  { el.innerHTML='<p class="file-empty">Set a files folder above.</p>'; return; }
  if (!data.valid)   { el.innerHTML='<p class="file-empty" style="color:#d12f2f">Folder path not found on disk.</p>'; return; }
  if (!data.files.length) { el.innerHTML='<p class="file-empty">Folder is empty.</p>'; return; }
  el.innerHTML = data.files.map(f=>`
    <div class="file-item">
      <span class="file-name">${f.name}</span>
      <span class="file-meta">${fmt(f.size)}</span>
    </div>`).join("");
}

function fmt(b){ return b<1024?b+" B":b<1048576?(b/1024).toFixed(1)+" KB":(b/1048576).toFixed(1)+" MB"; }

/* ── Embeddings ─────────────────────────────────────────────── */
const EMBED_DEFAULTS = { openai:"text-embedding-3-small", ollama:"nomic-embed-text" };

function updateEmbedPlaceholder() {
  const p = document.getElementById("embedProvider").value;
  document.getElementById("embedModel").placeholder = EMBED_DEFAULTS[p] || "model-name";
}

async function refreshEmbed() {
  const cfg = await fetch("/config").then(r=>r.json());
  document.getElementById("embedInfo").textContent = `${cfg.embed_provider} / ${cfg.embed_model}`;
  document.getElementById("embedInfo").style.color = "#1d1d1f";
  document.getElementById("embedProvider").value = cfg.embed_provider;
  document.getElementById("embedModel").value    = cfg.embed_model;
  updateEmbedPlaceholder();
  refreshIndexStatus();
}

function toggleEmbed() {
  const row = document.getElementById("embedEditRow");
  row.classList.contains("open") ? row.classList.remove("open") : (row.classList.add("open"), document.getElementById("embedModel").focus());
}

async function saveEmbed() {
  const provider = document.getElementById("embedProvider").value;
  const model    = document.getElementById("embedModel").value.trim() || EMBED_DEFAULTS[provider];
  await fetch("/keys",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({provider:"embed_provider",key:provider})});
  await fetch("/keys",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({provider:"embed_model",key:model})});
  embedMsg("Saved! Re-index files to apply the new model.",false);
  setTimeout(()=>{ document.getElementById("embedEditRow").classList.remove("open"); document.getElementById("embedMsg").textContent=""; refreshEmbed(); },2000);
}

function embedMsg(t,e){ const el=document.getElementById("embedMsg"); el.textContent=t; el.className="msg "+(e?"err":"ok"); }

/* ── Index ──────────────────────────────────────────────────── */
async function indexFiles() {
  const btn = document.getElementById("indexBtn");
  btn.disabled=true; btn.textContent="Indexing…";
  try {
    const r    = await fetch("/index",{method:"POST"});
    const data = await r.json();
    if (data.error) { alert("Error: "+data.error); }
    else {
      const ok  = data.indexed.length;
      const bad = data.failed.length;
      alert(`Done! ${ok} file(s) indexed (${data.total_chunks} chunks total)${bad?", "+bad+" failed":""}.`);
    }
    refreshIndexStatus();
  } catch(e) { alert("Request failed: "+e); }
  finally { btn.disabled=false; btn.textContent="Index All Files"; }
}

async function refreshIndexStatus() {
  const data = await fetch("/index/status").then(r=>r.json());
  const entries = Object.entries(data);
  document.getElementById("indexSummary").textContent =
    entries.length ? `${entries.length} file(s) indexed in ChromaDB` : "No files indexed yet";

  const el = document.getElementById("indexList");
  if (!entries.length) {
    el.innerHTML='<p class="file-empty">Click "Index All Files" to enable semantic search.</p>';
    return;
  }
  el.innerHTML = entries.map(([name,info])=>`
    <div class="file-item">
      <span class="file-name">${name}</span>
      <span class="file-meta">${info.chunks} chunks · ${info.provider}/${info.model}</span>
    </div>`).join("");
}

/* ── Init ───────────────────────────────────────────────────── */
async function checkPing() {
  try { const r=await fetch("/ping"); document.getElementById("pingDot").style.background=r.ok?"#30d158":"#ff453a"; }
  catch { document.getElementById("pingDot").style.background="#ff453a"; }
}

checkPing();
refreshKeys();
refreshFolder();
refreshEmbed();
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse, tags=["syftbox"])
def root():
    return HTMLResponse(_HTML)


@app.get("/ping", tags=["syftbox"])
def ping():
    return {"status": "ok", "app": app_name}
