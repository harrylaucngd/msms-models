#!/usr/bin/env bash
# Reproduce paper Table 3, rows (2) `chemformer_two_stage_inflated` and
# `chemformer_two_stage_corrected`. Run from the repository root after
# `bash paper/prepare_data.sh`, in the environment named in README.md. Needs a GPU.
set -euo pipefail
here=paper/table3/chemformer_two_stage
root=${OUT:-results/paper/table3}
for row in inflated corrected; do
  out=$root/chemformer_two_stage_$row
  msms-models predict --config "$here/config-$row.yaml" --output "$out" ${DEVICE:+--device "$DEVICE"}
  # 99.9% BCa intervals over 20,000 resamples. The archived intervals came from GLMR's own
  # bootstrap (NumPy default_rng(42 + metric index)), so these bounds differ in the last digit.
  msms-models evaluate --protocol table3 --seed 42 --queries "$out/queries.jsonl" \
    --predictions "$out/predictions.jsonl" --output "$out/metrics.json" --mces-cache "$out/mces.sqlite"
  # Means only, for the printed-value check (reuses the MCES cache).
  msms-models evaluate --protocol table3 --bootstrap-resamples 0 --queries "$out/queries.jsonl" \
    --predictions "$out/predictions.jsonl" --output "$out/metrics_means.json" --mces-cache "$out/mces.sqlite"
done
python paper/compare.py table3 chemformer_two_stage_inflated "$root/chemformer_two_stage_inflated/metrics_means.json"
python paper/compare.py table3 chemformer_two_stage_corrected "$root/chemformer_two_stage_corrected/metrics_means.json"
