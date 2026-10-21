#!/usr/bin/env bash
# Table 5, FRIGID-base. Run from the repository root after `bash paper/prepare_data.sh`.
set -euo pipefail
here=paper/table5/frigid_base
out=${OUT:-results/paper/table5/frigid_base}
msms-models predict --config "$here/config.yaml" --output "$out" ${DEVICE:+--device "$DEVICE"}
msms-models evaluate --protocol table5 --queries "$out/queries.jsonl" \
  --predictions "$out/predictions.jsonl" --output "$out/metrics.json" --mces-cache "$out/mces.sqlite"
