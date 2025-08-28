#!/usr/bin/env zsh
set -euo pipefail
set -o noglob
VENV_PY="${PWD}/.venv/bin/python"
if [ ! -x "$VENV_PY" ]; then
  echo "Python venv not found at $VENV_PY" >&2
  exit 1
fi
"$VENV_PY" -m pytest -q
