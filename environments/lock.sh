#!/usr/bin/env bash
# Regenerate environments/linux-cuda.txt, the one locked environment of every model and the
# evaluator (Linux x86-64, Python 3.11, PyTorch 2.7.0 with CUDA 12.8), from pyproject.toml.
# Run from anywhere; requires uv.
#
# Only torch is taken from the PyTorch wheel page and only DGL from DGL's; everything else
# comes from PyPI. docs/install.md explains the fixed versions; NumPy overrides
# bionemo-moco's numpy<2 because FRIGID's length model is a NumPy 2 pickle.
set -euo pipefail
cd "$(dirname "$0")/.."

uv pip compile pyproject.toml \
  --extra mces --extra mist --extra molforge --extra msnovelist --extra ms-gpt-base --extra frigid \
  --extra iceberg --extra glacier --extra dreams-chemberta --extra chemformer-two-stage \
  --extra chemberta --extra foam --extra deepsets --extra fingerprint-ffn \
  --extra random-generation --extra smiles-transformer \
  -c <(printf '%s\n' torch==2.7.0+cu128 dgl==1.1.3+cu121 scipy==1.15.3 pandas==2.3.3) \
  --override <(printf '%s\n' numpy==2.2.6) \
  --python-version 3.11 --python-platform x86_64-manylinux_2_28 \
  --find-links https://download.pytorch.org/whl/cu128/torch/ \
  --find-links https://data.dgl.ai/wheels/cu121/repo.html \
  --emit-index-url --emit-find-links --no-annotate \
  --custom-compile-command "bash environments/lock.sh" -o environments/linux-cuda.txt
