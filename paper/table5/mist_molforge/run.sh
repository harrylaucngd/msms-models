#!/usr/bin/env bash
# Tables 4 and 5, MIST 4M-aug + MolForge. Run from the repository root after `bash paper/prepare_data.sh`.
set -euo pipefail
here=paper/table5/mist_molforge
out=${OUT:-results/paper/table5/mist_molforge}
msms-models predict --config "$here/config.yaml" --output "$out" ${DEVICE:+--device "$DEVICE"}
msms-models evaluate --protocol table5 --queries "$out/queries.jsonl" \
  --predictions "$out/predictions.jsonl" --output "$out/metrics.json" --mces-cache "$out/mces.sqlite"
msms-models evaluate-fingerprints --config "$here/config.yaml" --queries "$out/queries.jsonl" \
  --output "$out/fingerprints.json"
