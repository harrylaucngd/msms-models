#!/usr/bin/env bash
# Table 3, DeepSets and DeepSets + Fourier features. Run from the repository root after `bash paper/prepare_data.sh`.
set -euo pipefail
here=paper/table3/deepsets
for variant in plain fourier; do
  out=${OUT:-results/paper/table3}/deepsets_$variant
  # The plain row predates Fourier features and MassSpecGym's m/z limit of 1005.
  flags=$([ "$variant" = plain ] && echo "--no-fourier-features --max-mz 1000" || true)
  [ -f "checkpoints/deepsets/$variant/model.pt" ] || python -m msms_models.models.retrieval.deepsets $flags \
    --train data/msg/train.jsonl --validation data/msg/val.jsonl \
    --output "checkpoints/deepsets/$variant" ${DEVICE:+--device "$DEVICE"}
  msms-models predict --config "$here/config-$variant.yaml" --output "$out" ${DEVICE:+--device "$DEVICE"}
  msms-models evaluate --protocol table3 --queries "$out/queries.jsonl" \
    --predictions "$out/predictions.jsonl" --output "$out/metrics.json" --mces-cache "$out/mces.sqlite"
done
