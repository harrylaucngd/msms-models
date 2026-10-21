#!/usr/bin/env bash
# Table 3, (1) DreaMS + ChemBERTa alignment. Run from the repository root after `bash paper/prepare_data.sh`.
set -euo pipefail
here=paper/table3/dreams_chemberta
# Candidate lists in SpecBridge's shuffled evaluation order, built from the official pools
# (the corrected row uses MassSpecGym 1.5, looked up by the RDKit-canonical target).
frozen=data/paper/table3/dreams_chemberta
[ -f "$frozen/candidates_inflated.jsonl" ] || python -m msms_models.models.retrieval.dreams_chemberta.candidates \
  --library data/MassSpecGym_retrieval_candidates_formula.json --queries data/msg/queries.jsonl \
  --output "$frozen/candidates_inflated.jsonl"
[ -f "$frozen/candidates_corrected.jsonl" ] || python -m msms_models.models.retrieval.dreams_chemberta.candidates \
  --library data/MassSpecGym1.5_retrieval_candidates_formula.json --queries data/msg/queries.jsonl \
  --output "$frozen/candidates_corrected.jsonl" --canonical-keys
for variant in inflated corrected; do
  out=${OUT:-results/paper/table3}/dreams_chemberta_$variant
  msms-models predict --config "$here/config-$variant.yaml" --output "$out" ${DEVICE:+--device "$DEVICE"}
  msms-models evaluate --protocol table3 --queries "$out/queries.jsonl" \
    --predictions "$out/predictions.jsonl" --output "$out/metrics.json" --mces-cache "$out/mces.sqlite"
done
