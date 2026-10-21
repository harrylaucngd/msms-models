#!/usr/bin/env bash
# Table 3, (2) Chemformer two-stage retrieval. Run from the repository root after `bash paper/prepare_data.sh`.
set -euo pipefail
here=paper/table3/chemformer_two_stage
for variant in inflated corrected; do
  out=${OUT:-results/paper/table3}/chemformer_two_stage_$variant
  msms-models predict --config "$here/config-$variant.yaml" --output "$out" ${DEVICE:+--device "$DEVICE"}
  msms-models evaluate --protocol table3 --queries "$out/queries.jsonl" \
    --predictions "$out/predictions.jsonl" --output "$out/metrics.json" --mces-cache "$out/mces.sqlite"
done
