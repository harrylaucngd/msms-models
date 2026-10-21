#!/usr/bin/env bash
# Reproduce paper Table 5, row `mist_molforge`. Run from the repository root after
# `bash paper/prepare_data.sh`, in the environment named in README.md.
set -euo pipefail
here=paper/table5/mist_molforge
out=${OUT:-results/paper/table5/mist_molforge}
msms-models predict --config "$here/config.yaml" --output "$out" ${DEVICE:+--device "$DEVICE"}
msms-models evaluate --protocol table5 --queries "$out/queries.jsonl" \
  --predictions "$out/predictions.jsonl" --output "$out/metrics.json" --mces-cache "$out/mces.sqlite"
python paper/compare.py table5 mist_molforge "$out/metrics.json"
