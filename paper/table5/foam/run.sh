#!/usr/bin/env bash
# Table 5, FOAM. Run from the repository root after `bash paper/prepare_data.sh` and building
# data/paper/table5/foam/seeds.jsonl with build_seeds.py (see README.md).
set -euo pipefail
here=paper/table5/foam
out=${OUT:-results/paper/table5/foam}
msms-models predict --config "$here/config.yaml" --output "$out" ${DEVICE:+--device "$DEVICE"}
msms-models evaluate --protocol table5 --queries "$out/queries.jsonl" \
  --predictions "$out/predictions.jsonl" --output "$out/metrics.json" --mces-cache "$out/mces.sqlite"
