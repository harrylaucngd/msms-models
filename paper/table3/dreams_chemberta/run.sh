#!/usr/bin/env bash
# Reproduce paper Table 3, rows `dreams_chemberta_inflated` and `dreams_chemberta_corrected`.
# Run from the repository root after `bash paper/prepare_data.sh`, in the environment named
# in README.md, with the two checkpoints in place.
set -euo pipefail
here=paper/table3/dreams_chemberta
frozen=data/paper/table3/dreams_chemberta
# Both candidate files store SpecBridge's shuffled evaluation order and are rebuilt from
# the official pools (sha256 pinned in the configs). The corrected row uses MassSpecGym 1.5,
# whose pool equals SpecBridge's `_ca` pool, looked up by the RDKit-canonical target.
build() {
  [ -f "$frozen/candidates_$1.jsonl" ] || python -m msms_models.models.retrieval.dreams_chemberta.candidates \
    --library "$2" --queries data/msg/queries.jsonl --output "$frozen/candidates_$1.jsonl" "${@:3}"
}
build inflated data/MassSpecGym_retrieval_candidates_formula.json
build corrected data/MassSpecGym1.5_retrieval_candidates_formula.json --canonical-keys
for row in inflated corrected; do
  out=${OUT:-results/paper/table3}/dreams_chemberta_$row
  msms-models predict --config "$here/config-$row.yaml" --output "$out" ${DEVICE:+--device "$DEVICE"}
  # Means only: the archived intervals used per-metric NumPy Generator seeds (README.md).
  msms-models evaluate --protocol table3 --bootstrap-resamples 0 --queries "$out/queries.jsonl" \
    --predictions "$out/predictions.jsonl" --output "$out/metrics.json" --mces-cache "$out/mces.sqlite"
  python paper/compare.py table3 "dreams_chemberta_$row" "$out/metrics.json"
done
