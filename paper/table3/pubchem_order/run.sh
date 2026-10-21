#!/usr/bin/env bash
# Reproduce paper Table 3, row `pubchem_order`. Run from the repository root after
# `bash paper/prepare_data.sh`. CPU only; no spectrum is read.
set -euo pipefail
here=paper/table3/pubchem_order
out=${OUT:-results/paper/table3/pubchem_order}
# Building the listing queries PubChem; the released frozen file is preferred.
[ -f data/paper/table3/pubchem_order.jsonl ] || python -m msms_models.models.retrieval.pubchem_order.build \
  --queries data/msg/queries.jsonl --output data/paper/table3/pubchem_order.jsonl
msms-models predict --config "$here/config.yaml" --output "$out"
msms-models evaluate --protocol table3 --queries "$out/queries.jsonl" \
  --predictions "$out/predictions.jsonl" --output "$out/metrics.json" --mces-cache "$out/mces.sqlite"
python paper/compare.py table3 pubchem_order "$out/metrics.json"
