#!/usr/bin/env bash
# Full reproduction: fresh venv, install, test, regenerate every number in
# README.md / claims.toml from scratch. No network auth required -- all four
# models are public, ungated Hugging Face checkpoints.
set -euo pipefail
cd "$(dirname "$0")/.."

python3.13 -m venv .venv 2>/dev/null || python3 -m venv .venv
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt pytest ruff

echo "=== lint ==="
ruff check .

echo "=== tests ==="
python -m pytest tests/ -q

echo "=== report (downloads ~3.5GB of model weights on first run) ==="
python scripts/gen_report.py
