#!/usr/bin/env bash
# Tables 4 and 5, MIST vDiffMS + MolForge at batch size 24 and 1. Run from the repository root after `bash paper/prepare_data.sh`.
set -euo pipefail
here=paper/table5/mist_vdiffms_molforge
# The 17,082-spectrum DiffMS cohort, in the original DataLoader order.
cohort=data/paper/table5/mist_vdiffms/queries.jsonl
if [ ! -f "$cohort" ]; then
  mkdir -p "$(dirname "$cohort")"
  python -c '
import json, sys
rows = {json.loads(line)["id"]: line for line in open(sys.argv[2])}
with open(sys.argv[3], "w") as out:
    out.writelines(rows[i.strip()] for i in open(sys.argv[1]) if i.strip())
' "$here/cohort_ids.txt" data/msg/queries.jsonl "$cohort"
fi
for variant in bs24 bs1; do
  out=${OUT:-results/paper/table5}/mist_vdiffms_molforge_$variant
  msms-models predict --config "$here/config-$variant.yaml" --output "$out" ${DEVICE:+--device "$DEVICE"}
  msms-models evaluate --protocol table5 --queries "$out/queries.jsonl" \
    --predictions "$out/predictions.jsonl" --output "$out/metrics.json" --mces-cache "$out/mces.sqlite"
done
