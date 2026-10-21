#!/usr/bin/env bash
# Reproduce paper Table 3, row `chiral_atoms`. Run from the repository root after
# `bash paper/prepare_data.sh`. CPU only; no spectrum is read.
set -euo pipefail
here=paper/table3/chiral_atoms
out=${OUT:-results/paper/table3/chiral_atoms}
# The ranking direction is fitted on the training split (printed for reference).
python -m msms_models.models.retrieval.chiral_atoms --queries data/msg/train.jsonl \
  --candidates data/MassSpecGym_retrieval_candidates_formula.json
msms-models predict --config "$here/config.yaml" --output "$out"
msms-models evaluate --protocol table3 --queries "$out/queries.jsonl" \
  --predictions "$out/predictions.jsonl" --output "$out/metrics.json" --mces-cache "$out/mces.sqlite"
python paper/compare.py table3 chiral_atoms "$out/metrics.json"
