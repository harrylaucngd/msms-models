"""Candidate fingerprints and declared, reproducible Tanimoto ranking."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from ..io import read_records


def load_candidates(path, queries):
    path = Path(path)
    if str(path).endswith(".json"):
        value = json.loads(path.read_text())
        if isinstance(value, dict):
            records = []
            for query in queries:
                # ID-keyed candidate libraries support target-free inference.
                # The official benchmark also provides a target-SMILES-keyed file.
                key = query["id"] if query["id"] in value else query.get("target_smiles")
                if key not in value:
                    raise ValueError(f"Candidate library is missing query {query['id']}")
                records.append({"id": query["id"], "candidates": value[key]})
        else:
            records = value
    else:
        records = read_records(path)
    lookup = {}
    for row in records:
        identifier = str(row.get("id", row.get("identifier", "")))
        if identifier in lookup:
            raise ValueError(f"Duplicate candidate record {identifier!r}")
        candidates = row.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"Query {identifier}: an ordered nonempty candidate list is required")
        lookup[identifier] = [c["smiles"] if isinstance(c, dict) else c for c in candidates]
    missing = [q["id"] for q in queries if q["id"] not in lookup]
    if missing:
        raise ValueError(f"Candidate library is missing queries: {missing[:5]}")
    return lookup


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
