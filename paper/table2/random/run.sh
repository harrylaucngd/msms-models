#!/usr/bin/env bash
# Table 2, Random and Random w/ ground-truth formula. Run from the repository root after `bash paper/prepare_data.sh`.
set -euo pipefail
here=paper/table2/random
for variant in mass mass-formula; do
  out=${OUT:-results/paper/table2}/random_${variant/-/_}
  msms-models predict --config "$here/config-$variant.yaml" --output "$out"
  msms-models evaluate --protocol table2 --queries "$out/queries.jsonl" \
    --predictions "$out/predictions.jsonl" --output "$out/metrics.json" --mces-cache "$out/mces.sqlite"
done
