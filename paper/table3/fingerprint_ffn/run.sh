#!/usr/bin/env bash
# Table 3, Fingerprint FFN. Run from the repository root after `bash paper/prepare_data.sh`.
set -euo pipefail
here=paper/table3/fingerprint_ffn
out=${OUT:-results/paper/table3/fingerprint_ffn}
[ -f checkpoints/fingerprint_ffn/model.pt ] || python -m msms_models.models.retrieval.fingerprint_ffn \
  --train data/msg/train.jsonl --validation data/msg/val.jsonl \
  --output checkpoints/fingerprint_ffn ${DEVICE:+--device "$DEVICE"}
msms-models predict --config "$here/config.yaml" --output "$out" ${DEVICE:+--device "$DEVICE"}
msms-models evaluate --protocol table3 --queries "$out/queries.jsonl" \
  --predictions "$out/predictions.jsonl" --output "$out/metrics.json" --mces-cache "$out/mces.sqlite"
