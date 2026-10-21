#!/usr/bin/env bash
# Table 3, (5) ranking by number of chiral atoms. Run from the repository root after `bash paper/prepare_data.sh`.
set -euo pipefail
here=paper/table3/chiral_atoms
out=${OUT:-results/paper/table3/chiral_atoms}
# The ranking direction in config.yaml was chosen on the training split with
#   python -m msms_models.models.retrieval.chiral_atoms --queries data/msg/train.jsonl \
#     --candidates data/MassSpecGym_retrieval_candidates_formula.json
msms-models predict --config "$here/config.yaml" --output "$out" ${DEVICE:+--device "$DEVICE"}
msms-models evaluate --protocol table3 --queries "$out/queries.jsonl" \
  --predictions "$out/predictions.jsonl" --output "$out/metrics.json" --mces-cache "$out/mces.sqlite"
