#!/usr/bin/env bash
# Table 5, Random generation. Run from the repository root after `bash paper/prepare_data.sh`.
set -euo pipefail
here=paper/table5/random_generation
out=${OUT:-results/paper/table5/random_generation}
[ -f checkpoints/random_generation/element_stats.pkl ] || \
  python -m msms_models.models.de_novo.random_generation.fit --train data/msg/train.jsonl \
    --output checkpoints/random_generation/element_stats.pkl
msms-models predict --config "$here/config.yaml" --output "$out"
msms-models evaluate --protocol table5 --queries "$out/queries.jsonl" \
  --predictions "$out/predictions.jsonl" --output "$out/metrics.json" --mces-cache "$out/mces.sqlite"
