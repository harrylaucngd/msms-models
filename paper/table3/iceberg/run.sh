#!/usr/bin/env bash
# Table 3, ICEBERG forward-model retrieval. Run from the repository root after `bash paper/prepare_data.sh`.
set -euo pipefail
here=paper/table3/iceberg
out=${OUT:-results/paper/table3/iceberg}
msms-models predict --config "$here/config.yaml" --output "$out" ${DEVICE:+--device "$DEVICE"}
msms-models evaluate --protocol table3 --queries "$out/queries.jsonl" \
  --predictions "$out/predictions.jsonl" --output "$out/metrics.json" --mces-cache "$out/mces.sqlite"
