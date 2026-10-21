#!/usr/bin/env bash
# Reproduce paper Table 3, row `nearest_neighbor`. Run from the repository root after
# `bash paper/prepare_data.sh`, in the environment named in README.md.
set -euo pipefail
here=paper/table3/nearest_neighbor
out=${OUT:-results/paper/table3/nearest_neighbor}
# The neighbour index is fitted on the training split; settings are chosen on validation.
[ -d data/indices/nearest-neighbor-msg ] || msms-models fit-nn \
  --train data/msg/train.jsonl --validation data/msg/val.jsonl --output data/indices/nearest-neighbor-msg
msms-models predict --config "$here/config.yaml" --output "$out" ${DEVICE:+--device "$DEVICE"}
msms-models evaluate --protocol table3 --queries "$out/queries.jsonl" \
  --predictions "$out/predictions.jsonl" --output "$out/metrics.json" --mces-cache "$out/mces.sqlite"
python paper/compare.py table3 nearest_neighbor "$out/metrics.json"
