#!/usr/bin/env bash
# Reproduce paper Table 3, rows `chemberta_classifier_inflated` and `_corrected`. Run from
# the repository root after `bash paper/prepare_data.sh`, with `.[chemberta,mces]`.
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
  python paper/compare.py table3 "chemberta_classifier_$variant" "$out/metrics.json"
done
