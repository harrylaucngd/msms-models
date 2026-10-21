"""Ordered candidate pools keyed by spectrum ID."""

from __future__ import annotations

import json
from pathlib import Path

from .io import read_records, record_id


def rdkit_smiles(smiles):
    from rdkit import Chem

    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


def load_candidates(path, queries):
    """Candidate SMILES per query ID, in the supplied order.

    ID-keyed files are read directly. The official JSON files are keyed by the target
    SMILES; the 1.5 files by its RDKit canonical form.
    """
    path = Path(path)
    if path.suffix == ".json":
        library = json.loads(path.read_text())
        lookup = {}
        for query in queries:
            key = query["id"] if query["id"] in library else query["target_smiles"]
            lookup[query["id"]] = library[key] if key in library else library[rdkit_smiles(key)]
        return lookup
    return {
        record_id(row): [c["smiles"] if isinstance(c, dict) else c for c in row["candidates"]]
        for row in read_records(path)
    }
