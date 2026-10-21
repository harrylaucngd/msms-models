"""Rank candidates by Tanimoto similarity to a predicted Morgan4096 fingerprint."""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


def rank_fingerprints(queries, probabilities, candidates, *, threshold=None, ties="stable"):
    """Continuous Tanimoto against each candidate's Morgan r2/4096 fingerprint.

    ``ties="legacy-quicksort"`` reproduces NumPy's reverse quicksort, the tie order of the
    original MIST evaluation; ``"stable"`` keeps the supplied candidate order on ties.
    """
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if threshold is not None:
        probabilities = (probabilities >= threshold).astype(np.float32)
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=4096, includeChirality=False
    )
    cache, rows = {}, []
    for query, vector in zip(queries, probabilities, strict=True):
        strings = candidates[query["id"]]
        for smiles in strings:
            if smiles not in cache:
                fingerprint = generator.GetFingerprintAsNumPy(Chem.MolFromSmiles(smiles))
                cache[smiles] = np.packbits(fingerprint, bitorder="little")
        matrix = np.unpackbits(
            np.stack([cache[s] for s in strings]), axis=1, bitorder="little"
        ).astype(np.float32)
        intersection = matrix @ vector
        scores = intersection / np.maximum(vector.sum() + matrix.sum(1) - intersection, 1e-8)
        order = np.argsort(-scores, kind="stable") if ties == "stable" else np.argsort(scores)[::-1]
        rows.append({"id": query["id"], "candidates": [strings[i] for i in order],
                     "scores": scores[order].astype(float).tolist(), "status": "ok"})
    return rows
