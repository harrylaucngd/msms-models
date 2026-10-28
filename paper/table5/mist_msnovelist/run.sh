#!/usr/bin/env bash
# Table 5, MIST + MSNovelist. Run from the repository root after `bash paper/prepare_data.sh`
# and training the decoder (see README.md).
set -euo pipefail
here=paper/table5/mist_msnovelist
msms-models prepare --input data/msg/queries.jsonl --output data/paper/table5/mist_vdiffms/queries.jsonl \
  --ids paper/table5/mist_vdiffms_molforge/cohort_ids.txt
out=${OUT:-results/paper/table5/mist_msnovelist}
msms-models predict --config "$here/config.yaml" --output "$out" ${DEVICE:+--device "$DEVICE"}
msms-models evaluate --protocol table5 --queries "$out/queries.jsonl" \
  --predictions "$out/predictions.jsonl" --output "$out/metrics.json" --mces-cache "$out/mces.sqlite"
