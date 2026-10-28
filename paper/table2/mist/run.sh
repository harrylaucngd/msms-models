#!/usr/bin/env bash
# Table 2, MIST with MIST-CF or ground-truth formulas. Run from the repository root after `bash paper/prepare_data.sh`.
set -euo pipefail
here=paper/table2/mist
for row in mistcf_standard mistcf_testonly mistcf_traintest ground_truth; do
  # The spectra with a formula: those with a posterior.
  msms-models prepare --input data/msg/queries.jsonl --output "data/paper/table2/queries_$row.jsonl" \
    --ids "data/paper/posteriors/mist_4m_aug_table2_$row/query_ids.npy"
  out=${OUT:-results/paper/table2}/mist_$row
  msms-models predict --config "$here/config-$row.yaml" --output "$out"
  msms-models evaluate --protocol table2 --queries "$out/queries.jsonl" \
    --predictions "$out/predictions.jsonl" --output "$out/metrics.json" --mces-cache "$out/mces.sqlite"
done
