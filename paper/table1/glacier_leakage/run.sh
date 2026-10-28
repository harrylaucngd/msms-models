#!/usr/bin/env bash
# Table 1, settings 1-5. Run from the repository root after `bash paper/prepare_data.sh`.
set -euo pipefail
here=paper/table1/glacier_leakage
for setting in 1 2 3 4 5; do
  config=$here/config-setting$setting.yaml
  out=${OUT:-results/paper/table1}/setting$setting
  msms-models predict --config "$config" --output "$out"
  msms-models evaluate --protocol table3 --queries "$out/queries.jsonl" \
    --predictions "$out/predictions.jsonl" --output "$out/metrics.json" --mces-cache "$out/mces.sqlite"
  msms-models evaluate-fingerprints --config "$config" --queries "$out/queries.jsonl" \
    --output "$out/fingerprints.json"
done
