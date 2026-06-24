# Dare Workflow Runner (SyftBox app)

A standalone SyftBox app that executes **Dare workflows** without Dare's engine.
Given a workflow id, it pulls the workflow's self-contained graph from Dare, runs
it node-by-node, and delegates all file access + retrieval back to Dare's APIs.
The LLM is called directly by the runner.

- **SyftBox = execution engine. Dare = data layer (files + retrieval).**
- No embedding/indexing/storage happens here — those stay in Dare.

## What it does

```
POST /run  { "workflowId": 3 }
   │
   ├─▶ GET  {DARE}/api/workflows/{id}/export/     → workflow graph (JSON)
   │
   └─ execute nodes ─┬─ File(content)    → GET  {DARE}/api/files/{id}/content/
                     ├─ File(embeddings) → POST {DARE}/api/retrieval/query/   (vector search)
                     └─ Step / Router    → LLM provider (OpenAI / Claude / Gemini / Ollama)
   │
   ▼
returns the run result + saves it to your datasite under app_data/<app>/runs/
```

Supported node types: `start`, `step`, `file`, `router`, `output` (`notes` ignored).

## Setup

1. Place this folder at `~/SyftBox/apps/dare_workflow_runner/` — the SyftBox
   daemon auto-detects and runs it (via `run.sh`).
2. Create your credentials file:
   ```bash
   cp .env.example .env
   ```
   Fill in `.env` with **your own** values:
   - `DARE_TOKEN` — a fresh access token from the Dare login endpoint
   - an LLM key — `OPENAI_API_KEY` (and/or `CLAUDE_API_KEY`, `GEMINI_API_KEY`)
3. Make sure the **Dare backend is running** (it serves the export + data APIs).

The app listens on a fixed local port **8081** (override with `RUNNER_PORT`).

## Run a workflow

Send a request to the `/run` endpoint:

```bash
curl -X POST http://localhost:8081/run \
  -H "Content-Type: application/json" \
  -d '{ "workflowId": 3 }'
```

- Pass `{ "workflowId": <id> }` to have the app pull the export from Dare, **or**
  `{ "workflow": { ...exported JSON... } }` to run an inline graph.
- The response contains each node's status/output and a `saved_to` path.
- Health check: `GET http://localhost:8081/ping`.

### Local testing without SyftBox

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_local.py sample_export.json          # run a saved export
python run_local.py --workflow-id 3             # or pull from Dare by id
```

## Project layout

| Path | Purpose |
|------|---------|
| `main.py` | SyftBox app — exposes `/run` and `/ping`, saves results to the datasite |
| `run_local.py` | CLI for local testing (no SyftBox needed) |
| `run.sh` | startup script the SyftBox daemon uses |
| `workflow_loader.py` | validates the export JSON, builds the graph |
| `graph.py` | node/edge graph + topological ordering |
| `execution_engine.py` | the execution loop (ordering, routing) |
| `state_manager.py` | tracks per-node output/status/history |
| `prompts.py` | builds the LLM prompt (instructions + context + task) |
| `services.py` | bundles the Dare + LLM clients |
| `clients/dare_client.py` | calls Dare: export, file content, retrieval |
| `clients/llm_client.py` | calls the LLM providers (OpenAI/Claude/Gemini/Ollama) |
| `handlers/` | one handler per node type (start/step/file/router/output) |

## Notes

- `DARE_TOKEN` expires (~12h) — refresh it from the Dare login endpoint.
- Claude/Gemini/Ollama paths are implemented; OpenAI is the most tested.
- Credentials live only in `.env` (gitignored) — never commit them.
