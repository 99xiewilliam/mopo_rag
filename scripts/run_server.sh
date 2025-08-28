#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ -f "${ROOT_DIR}/.venv/bin/activate" ]; then
  source "${ROOT_DIR}/.venv/bin/activate"
fi

exec uvicorn apps.api.main:app --host 0.0.0.0 --port 8000


