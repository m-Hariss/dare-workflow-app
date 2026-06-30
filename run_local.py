"""Local CLI for testing the engine without SyftBox.

Usage:
    python run_local.py sample_export.json
    python run_local.py sample_export.json --output result.json

The workflow JSON is loaded from a local file — exported once from Dare and
saved to disk. No network calls to Dare are made at runtime.
"""
import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from app_state import make_services
from core.engine import ExecutionEngine
from core.loader import load_workflow

load_dotenv(Path(__file__).resolve().parent / ".env")


def main():
    parser = argparse.ArgumentParser(description="Run an exported workflow JSON locally.")
    parser.add_argument("source", help="Path to exported workflow JSON file")
    parser.add_argument("--output", help="Write the result JSON to this path")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        print(f"Error: file not found: {args.source}", file=sys.stderr)
        sys.exit(1)

    payload = json.loads(source.read_text())
    graph   = load_workflow(payload)
    result  = ExecutionEngine(graph, make_services()).run()

    rendered = json.dumps(result, indent=2, default=str)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered)

    sys.exit(0 if result["status"] == "completed" else 1)


if __name__ == "__main__":
    main()
