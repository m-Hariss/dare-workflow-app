"""Local CLI for testing the engine without going through SyftBox RPC.

Usage:
    # run from a saved export file
    python run_local.py sample_export.json

    # or pull the export straight from Dare by id
    DARE_TOKEN=... python run_local.py --workflow-id 1

Environment:
    DARE_BASE_URL   (default http://localhost:8000)
    DARE_TOKEN      Bearer token for Dare APIs
    OPENAI_API_KEY  key for the LLM provider used by the workflow
"""
import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from execution_engine import ExecutionEngine
from services import Services
from workflow_loader import load_workflow

# Load .env so the CLI picks up the same credentials as the SyftBox app.
load_dotenv(Path(__file__).resolve().parent / ".env")


def main():
    parser = argparse.ArgumentParser(description="Run an exported Dare workflow locally.")
    parser.add_argument("source", nargs="?", help="Path to exported workflow JSON")
    parser.add_argument("--workflow-id", type=int, help="Fetch the export from Dare by id")
    parser.add_argument("--output", help="Write the result JSON to this path")
    args = parser.parse_args()

    services = Services.from_config()

    if args.workflow_id:
        payload = services.dare.export_workflow(args.workflow_id)
    elif args.source:
        with open(args.source) as f:
            payload = json.load(f)
    else:
        parser.error("provide a JSON file path or --workflow-id")
        return

    graph = load_workflow(payload)
    result = ExecutionEngine(graph, services).run()

    rendered = json.dumps(result, indent=2, default=str)
    print(rendered)
    if args.output:
        with open(args.output, "w") as f:
            f.write(rendered)

    sys.exit(0 if result["status"] == "completed" else 1)


if __name__ == "__main__":
    main()
