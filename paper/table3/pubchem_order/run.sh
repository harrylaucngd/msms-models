#!/usr/bin/env bash
# Table 3, (4) PubChem default ranking. Run from the repository root after `bash paper/prepare_data.sh`.
set -euo pipefail
here=paper/table3/pubchem_order
out=${OUT:-results/paper/table3/pubchem_order}
# Freeze PubChem's formula listings (needs internet access) unless they are already present.
[ -f data/paper/table3/pubchem_order.jsonl ] || python -m msms_models.models.retrieval.pubchem_order.build \
  --queries data/msg/queries.jsonl --output data/paper/table3/pubchem_order.jsonl
msms-models predict --config "$here/config.yaml" --output "$out" ${DEVICE:+--device "$DEVICE"}
msms-models evaluate --protocol table3 --queries "$out/queries.jsonl" \
  --predictions "$out/predictions.jsonl" --output "$out/metrics.json" --mces-cache "$out/mces.sqlite"
