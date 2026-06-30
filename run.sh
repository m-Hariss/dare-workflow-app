#!/bin/sh

set -e

if [ ! -d .venv ]; then
  uv venv
fi

. .venv/bin/activate

uv pip install --upgrade -r requirements.txt

echo "Running SyftBox app with $(python3 --version) at $(which python3)"

# Pin to a fixed local HTTP port so the testing URL stays constant.
# (SyftBox syft:// RPC works via the filesystem rpc/ folder, not this port,
#  so ignoring SYFTBOX_ASSIGNED_PORT is safe.)
uvicorn main:app --host 0.0.0.0 --port ${RUNNER_PORT:-8081} --reload \
  --reload-dir core \
  --reload-dir api \
  --reload-dir handlers \
  --reload-dir providers \
  --reload-dir rag \
  --reload-dir storage \
  --reload-dir web

deactivate
