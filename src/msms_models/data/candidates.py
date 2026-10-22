"""Ordered candidate pools keyed by spectrum ID."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .io import read_records, record_id


def rdkit_smiles(smiles):
    from rdkit import Chem

    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


def inchikey14(smiles):
    from rdkit import Chem

    return Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))[:14]


def load_candidates(path, queries):
    """Candidate SMILES per query ID, in the supplied order.

    ID-keyed files are read directly. The official JSON files are keyed by the target
    SMILES; the 1.5 files by its RDKit canonical form. Newer RDKit releases write a few
    stereo SMILES differently from the one that built the 1.5 files, so a target found
    under neither key is matched by the first InChIKey block, when exactly one key has it.
    """
    path = Path(path)
    if path.suffix == ".json":
        library = json.loads(path.read_text())
        skeletons = None
        lookup = {}
        for query in queries:
            key = query["id"] if query["id"] in library else query["target_smiles"]
            if key not in library:
                key = rdkit_smiles(key)
            if key not in library:
                if skeletons is None:
                    skeletons = defaultdict(list)
                    for smiles in library:
                        skeletons[inchikey14(smiles)].append(smiles)
                matches = skeletons[inchikey14(query["target_smiles"])]
                if len(matches) != 1:
                    raise KeyError(f"No unique candidate list for {query['target_smiles']!r}")
                key = matches[0]
            lookup[query["id"]] = library[key]
        return lookup
    return {
        record_id(row): [c["smiles"] if isinstance(c, dict) else c for c in row["candidates"]]
        for row in read_records(path)
    }
