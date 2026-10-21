#!/usr/bin/env bash
# Reproduce the paper's Table 5 diagnostic rows mist_vdiffms_molforge_bs24 and _bs1
# (Table 4: MIST vDiffMS + MolForge). Run from the repository root after
# `bash paper/prepare_data.sh`, in the environment named in README.md.
set -euo pipefail
here=paper/table5/mist_vdiffms_molforge
cohort=data/paper/table5/mist_vdiffms/queries.jsonl
# The 17,082-spectrum cohort, in the original DataLoader order, selected from the
# official test queries by the frozen ID list (see README.md for how it was built).
if [ ! -f "$cohort" ]; then
  mkdir -p "$(dirname "$cohort")"
  python - "$here/cohort_ids.txt" data/msg/queries.jsonl "$cohort" <<'EOF'
import hashlib, json, sys
ids = [line.strip() for line in open(sys.argv[1]) if line.strip()]
rows = {}
for line in open(sys.argv[2]):
    row = json.loads(line)
    rows[row["id"]] = line if line.endswith("\n") else line + "\n"
missing = [i for i in ids if i not in rows]
if len(ids) != 17082 or len(set(ids)) != len(ids) or missing:
    sys.exit(f"cohort ids invalid or missing from queries: {missing[:5]}")
text = "".join(rows[i] for i in ids)
expected = "319992d53cb9eeba602d96e46193f4fb57eef9ad5074fab9c5eeabc212835f40"
if hashlib.sha256(text.encode()).hexdigest() != expected:
    sys.exit("cohort queries differ from the frozen manifest; rerun paper/prepare_data.sh")
with open(sys.argv[3], "w") as out:
    out.write(text)
EOF
fi
status=0
for variant in bs24 bs1; do
  out=${OUT:-results/paper/table5}/mist_vdiffms_molforge_$variant
  msms-models predict --config "$here/config-$variant.yaml" --output "$out" \
    ${DEVICE:+--device "$DEVICE"}
  msms-models evaluate --protocol table5 --queries "$out/queries.jsonl" \
    --predictions "$out/predictions.jsonl" --output "$out/metrics.json" \
    --mces-cache "$out/mces.sqlite"
  python paper/compare.py table5 "mist_vdiffms_molforge_$variant" "$out/metrics.json" || status=1
done
exit $status
