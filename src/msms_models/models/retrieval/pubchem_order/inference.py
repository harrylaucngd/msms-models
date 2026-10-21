"""Rank each formula's candidates by their position in a frozen PubChem formula listing.

Paper Appendix A.4, method 4. PubChem returns the compounds matching a formula in a
default order that favours well-studied molecules; ranking candidates by that order
ignores the spectrum. The listing is frozen by ``build.py`` so inference is offline.
"""

from __future__ import annotations

import numpy as np

from ....data.candidates import load_candidates
from ....data.io import read_records
from ....evaluation.chemistry import inchikey14


def predict(config, queries):
    paths = config.require_assets("candidates", "pubchem_order")
    candidates = load_candidates(paths["candidates"], queries)
    listings = {r["formula"]: r["inchikey14"] for r in read_records(paths["pubchem_order"])}
    rng = np.random.default_rng(config.seed)
    rows = []
    for query in queries:
        position = {}
        for i, key in enumerate(listings[query["formula"]]):
            position.setdefault(key, i)
        strings = candidates[query["id"]]
        ranks = [position.get(inchikey14(s), len(position)) for s in strings]
        # Unlisted candidates tie; break ties randomly, because the official pools list
        # the true molecule first and a stable order would reveal it.
        order = np.lexsort((rng.permutation(len(strings)), ranks)).tolist()
        rows.append({
            "id": query["id"], "candidates": [strings[i] for i in order],
            "scores": [-float(ranks[i]) for i in order], "status": "ok",
        })
    return rows
