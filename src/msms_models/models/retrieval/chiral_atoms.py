"""Spectrum-blind baseline: rank candidates by their number of annotated stereocentres.

Paper Appendix A.4, method 5. The count is the number of atoms whose chiral tag is
set in the candidate's SMILES. The ranking direction is chosen on the training
split: descending if true molecules carry more annotated stereocentres than their
candidates on average, ascending otherwise. No spectrum is read.

    python -m msms_models.models.retrieval.chiral_atoms \\
        --queries data/msg/train.jsonl --candidates data/MassSpecGym_retrieval_candidates_formula.json
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache

import numpy as np

from ...data.candidates import load_candidates
from ...data.io import atomic_json, load_queries


@lru_cache(maxsize=500_000)
def chiral_atoms(smiles: str) -> int:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return sum(a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for a in mol.GetAtoms())


def fit_direction(queries, candidates) -> dict:
    """Compare true molecules with their own candidates; uses training targets only."""
    true = [chiral_atoms(q["target_smiles"]) for q in queries]
    decoys = [np.mean([chiral_atoms(c) for c in candidates[q["id"]]]) for q in queries]
    direction = "descending" if np.mean(true) > np.mean(decoys) else "ascending"
    return {"direction": direction, "mean_true": float(np.mean(true)),
            "mean_candidates": float(np.mean(decoys)), "queries": len(queries)}


def validate(config, queries):
    if config.parameters.get("direction") not in {"descending", "ascending"}:
        raise ValueError("Set direction (descending or ascending) from fit_direction on training data")
    if config.parameters.get("ties", "random") not in {"random", "stable"}:
        raise ValueError("ties must be random or stable")


def predict(config, queries):
    paths = config.require_assets("candidates")
    candidates = load_candidates(paths["candidates"], queries)
    sign = -1 if config.parameters["direction"] == "descending" else 1
    ties = config.parameters.get("ties", "random")
    rng = np.random.default_rng(config.seed)
    rows = []
    for query in queries:
        strings = candidates[query["id"]]
        counts = np.array([chiral_atoms(s) for s in strings])
        # Random tie-breaking keeps the supplied candidate order, which can place the
        # true molecule first, from deciding ties.
        tiebreak = rng.permutation(len(strings)) if ties == "random" else np.arange(len(strings))
        order = np.lexsort((tiebreak, sign * counts))
        rows.append({
            "id": query["id"], "candidates": [strings[i] for i in order],
            "scores": [float(-sign * counts[i]) for i in order], "status": "ok",
        })
    atomic_json(config.output / "ranking_protocol.json", {
        "score": "number of atoms with a chiral tag", "direction": config.parameters["direction"],
        "ties": ties, "seed": config.seed, "spectrum_used": False,
    })
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description="Choose the ranking direction on training data")
    parser.add_argument("--queries", required=True, help="Training queries with target_smiles")
    parser.add_argument("--candidates", required=True, help="Candidate pool covering those queries")
    args = parser.parse_args(argv)
    queries = load_queries(args.queries, require_target=True)
    print(json.dumps(fit_direction(queries, load_candidates(args.candidates, queries)), indent=2))


if __name__ == "__main__":
    main()
