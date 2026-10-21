#!/usr/bin/env bash
# Table 3, Nearest Neighbor. Run from the repository root after `bash paper/prepare_data.sh`.
set -euo pipefail
here=paper/table3/nearest_neighbor
out=${OUT:-results/paper/table3/nearest_neighbor}
# Fit the neighbour index on training spectra and choose its settings on validation.
[ -d data/indices/nearest-neighbor-msg ] || msms-models fit-nn \
  --train data/msg/train.jsonl --validation data/msg/val.jsonl --output data/indices/nearest-neighbor-msg
msms-models predict --config "$here/config.yaml" --output "$out" ${DEVICE:+--device "$DEVICE"}
msms-models evaluate --protocol table3 --queries "$out/queries.jsonl" \
  --predictions "$out/predictions.jsonl" --output "$out/metrics.json" --mces-cache "$out/mces.sqlite"
