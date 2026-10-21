#!/usr/bin/env bash
# Table 3, (3) ChemBERTa SMILES classifier. Run from the repository root after `bash paper/prepare_data.sh`.
set -euo pipefail
here=paper/table3/chemberta_classifier
for variant in inflated corrected; do
  out=${OUT:-results/paper/table3}/chemberta_classifier_$variant
  flag=$([ "$variant" = corrected ] && echo --canonical || true)
  [ -d "checkpoints/chemberta_classifier/$variant" ] || \
    python -m msms_models.models.retrieval.chemberta_classifier.train $flag \
      --queries data/msg/train.jsonl --candidates data/MassSpecGym_retrieval_candidates_formula.json \
      --output "checkpoints/chemberta_classifier/$variant" ${DEVICE:+--device "$DEVICE"}
  msms-models predict --config "$here/config-$variant.yaml" --output "$out" ${DEVICE:+--device "$DEVICE"}
  msms-models evaluate --protocol table3 --queries "$out/queries.jsonl" \
    --predictions "$out/predictions.jsonl" --output "$out/metrics.json" --mces-cache "$out/mces.sqlite"
done
