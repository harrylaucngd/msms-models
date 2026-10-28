#!/usr/bin/env bash
# Tables 4 and 5, MIST vDiffMS + MolForge at batch size 24 and 1. Run from the repository root after `bash paper/prepare_data.sh`.
set -euo pipefail
here=paper/table5/mist_vdiffms_molforge
# The 17,082 spectra of the DiffMS evaluation, in its order.
msms-models prepare --input data/msg/queries.jsonl --output data/paper/table5/mist_vdiffms/queries.jsonl \
  --ids "$here/cohort_ids.txt"
for variant in bs24 bs1; do
  out=${OUT:-results/paper/table5}/mist_vdiffms_molforge_$variant
  msms-models predict --config "$here/config-$variant.yaml" --output "$out" ${DEVICE:+--device "$DEVICE"}
  msms-models evaluate --protocol table5 --queries "$out/queries.jsonl" \
    --predictions "$out/predictions.jsonl" --output "$out/metrics.json" --mces-cache "$out/mces.sqlite"
  msms-models evaluate-fingerprints --config "$here/config-$variant.yaml" --queries "$out/queries.jsonl" \
    --posterior "$out/posterior" --output "$out/fingerprints.json"
done
