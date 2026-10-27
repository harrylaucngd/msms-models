#!/usr/bin/env bash
# Prepare the official MassSpecGym inputs shared by the paper rows. Run from the repository root.
# Frozen inputs that cannot be derived from the official release (imputed collision
# energies, the Table 3 forward-model candidate pool, MIST posterior caches) are
# downloaded separately into data/paper/; see paper/README.md.
set -euo pipefail
msms-models download msg --output data --candidates formula
for split in train val; do
  msms-models prepare --input data/MassSpecGym.tsv --split "$split" --formula-from-target \
    --output "data/msg/$split.jsonl"
done
# The "bonus" challenge: the molecular formula of every test spectrum is known.
msms-models prepare --input data/MassSpecGym.tsv --split test --formula-from-target \
  --output data/msg/queries.jsonl
msms-models prepare-candidates --input data/MassSpecGym_retrieval_candidates_formula.json \
  --queries data/msg/queries.jsonl --output data/msg/candidates_formula.jsonl
# MassSpecGym 1.5: the same pool with every SMILES RDKit-canonicalized ("corrected" rows).
msms-models download msg --output data --candidates formula --candidate-version 1.5
msms-models prepare-candidates --input data/MassSpecGym1.5_retrieval_candidates_formula.json \
  --queries data/msg/queries.jsonl --output data/msg/candidates_formula_1.5.jsonl
# Table 2: the mass-based retrieval pool.
msms-models download msg --output data --candidates mass
msms-models prepare-candidates --input data/MassSpecGym_retrieval_candidates_mass.json \
  --queries data/msg/queries.jsonl --output data/msg/candidates_mass.jsonl
