#!/usr/bin/env bash
# Create the locked environment and install msms-models into it (Linux x86-64, CUDA 12.8):
#
#   bash environments/install.sh [.venv]
#
# Requires uv (https://docs.astral.sh/uv/).
set -euo pipefail
venv=${1:-.venv}
root=$(cd "$(dirname "$0")/.." && pwd)

uv venv --python 3.11 "$venv"
uv pip sync --python "$venv/bin/python" "$root/environments/linux-cuda.txt"
uv pip install --python "$venv/bin/python" --no-deps -e "$root"
echo "Activate with: source $venv/bin/activate"
