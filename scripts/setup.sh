#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# MLX Workbench — local setup (macOS / Apple Silicon).
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "==> MLX Workbench setup"
echo "    Project root: $PROJECT_ROOT"

command -v uv   >/dev/null || { echo "ERROR: uv not found (https://docs.astral.sh/uv/)"; exit 1; }
command -v node >/dev/null || { echo "ERROR: node not found"; exit 1; }
command -v npm  >/dev/null || { echo "ERROR: npm not found"; exit 1; }

if [[ "$(uname -m)" != "arm64" || "$(uname -s)" != "Darwin" ]]; then
  echo "WARNING: MLX requires Apple Silicon macOS — backend will fail to load models on this host."
fi

echo "==> Installing Python deps via uv"
uv sync

echo "==> Installing frontend deps"
( cd frontend && npm install --silent )

echo
echo "Setup complete."
echo
echo "  Backend:   cd backend && ../.venv/bin/python -m uvicorn main:app --reload --port 8000"
echo "  Frontend:  cd frontend && npm run dev"
echo "  API docs:  http://localhost:8000/docs"
echo "  UI:        http://localhost:3000"
