"""Random retrieval, a MassSpecGym baseline (Table 2: Random, Random w/ ground-truth formula; Table 3: Random).

Candidates are ranked by uniform random scores (``RandomRetrieval``). With
``formula_filter``, only candidates whose molecular formula equals the query's are scored,
and the others follow them (``RandomRetrievalGTFormula``); on the mass-based challenge that
gives the formula-known pool. No spectrum is read.
"""

from __future__ import annotations

import numpy as np

from ...data.candidates import load_candidates


def candidate_formula(smiles):
    from rdkit import Chem
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula

    mol = Chem.MolFromSmiles(smiles)
    return CalcMolFormula(mol) if mol is not None else ""


def predict(config, queries):
    candidates = load_candidates(config.require_assets("candidates")["candidates"], queries)
    formula_filter = config.parameters.get("formula_filter", False)
    rng = np.random.default_rng(config.seed)
    rows = []
    for query in queries:
        strings = candidates[query["id"]]
        scores = rng.random(len(strings))
        if formula_filter:
            matching = np.array([candidate_formula(s) == query["formula"] for s in strings])
            scores = np.where(matching, scores + 1, scores) - 1  # non-matching candidates last
        order = np.argsort(-scores, kind="stable")
        rows.append({"id": query["id"], "candidates": [strings[i] for i in order],
                     "scores": scores[order].tolist(), "status": "ok"})
    return rows
