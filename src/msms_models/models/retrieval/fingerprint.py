"""Rank candidates by Tanimoto similarity to a predicted Morgan4096 fingerprint."""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


def rank_fingerprints(queries, probabilities, candidates, *, threshold=None, ties="stable"):
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if probabilities.shape != (len(queries), 4096):
        raise ValueError("Expected one Morgan4096 prediction per query")
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1.000001).any()
    ):
        raise ValueError("Invalid fingerprint probabilities")
    if threshold is not None:
        if not 0 <= threshold <= 1:
            raise ValueError("Fingerprint threshold must lie in [0, 1]")
        probabilities = (probabilities >= threshold).astype(np.float32)
    if ties not in {"stable", "legacy-quicksort"}:
        raise ValueError("Unknown candidate tie policy")
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=4096, includeChirality=False
    )
    cache = {}
    rows = []
    for query, vector in zip(queries, probabilities, strict=True):
        strings = candidates[query["id"]]
        for smiles in strings:
            if smiles not in cache:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    raise ValueError(f"Invalid candidate for {query['id']}: {smiles!r}")
                cache[smiles] = np.packbits(generator.GetFingerprintAsNumPy(mol), bitorder="little")
        matrix = np.unpackbits(
            np.stack([cache[s] for s in strings]), axis=1, bitorder="little"
        ).astype(np.float32)
        intersection = matrix @ vector
        scores = intersection / np.maximum(vector.sum() + matrix.sum(1) - intersection, 1e-8)
        order = np.argsort(-scores, kind="stable") if ties == "stable" else np.argsort(scores)[::-1]
        rows.append(
            {
                "id": query["id"],
                "candidates": [strings[i] for i in order],
                "scores": scores[order].astype(float).tolist(),
                "status": "ok",
            }
        )
    return rows
