"""Candidate retrieval by forward spectral prediction and native entropy distance."""

from __future__ import annotations

import numpy as np
from rdkit import Chem

from ..data.candidates import load_candidates
from .vendor.spectral_scoring import _entropy_sparse, bin_spectrum, expand_scores_to_full_candidates

# The chemical domain of the ms-pred forward models.
ELEMENTS = {"C", "N", "P", "O", "S", "Si", "I", "H", "Cl", "F", "Br", "B", "Se", "Fe", "Co", "As",
            "Na", "K"}


def canonical_candidate(smiles, *, max_heavy_atoms=10000):
    """ms-pred's input normalization (``prepare_entry``), or None outside the model domain."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or any(a.GetSymbol() not in ELEMENTS for a in mol.GetAtoms()):
        return None
    mol = Chem.RemoveHs(mol)
    if mol.GetNumHeavyAtoms() > max_heavy_atoms:
        return None
    return Chem.MolToSmiles(mol), Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))


def spectrum_settings(config):
    p = config.parameters
    return int(p.get("num_bins", 150000)), float(p.get("upper_limit", 1500)), int(p.get("sparse_k", 100))


def score_candidates(predictions, target):
    """Native single-energy aggregation, including its normalization epsilon."""
    distances = _entropy_sparse([{0: p} for p in predictions], target, 0, sqrt_inten=True)
    n = int(np.count_nonzero(target[1] > 0))
    weight = int(n >= 5) * 3 + int(n >= 1)
    return distances * (np.float64(weight) / (weight + 1e-9))


def retrieve(config, queries, forward):
    """Rank each query's candidates by entropy distance to spectra predicted by ``forward``.

    ``query_intensity_space: native`` takes the query peaks as prepared; ``raw`` applies a
    square root first. Candidates outside the forward model's domain get distance 1.
    """
    candidates = load_candidates(config.require_assets("candidates")["candidates"], queries)
    raw_intensities = config.parameters.get("query_intensity_space", "native") == "raw"
    max_heavy_atoms = int(config.parameters.get("max_heavy_atoms", 10000))
    bins, upper, _ = spectrum_settings(config)
    rows = []
    for query in queries:
        peaks = np.column_stack([query["spectrum"]["mz"], query["spectrum"]["intensity"]])
        if raw_intensities:
            peaks[:, 1] = np.sqrt(peaks[:, 1])
        target = bin_spectrum(peaks, bins, upper, pool_fn="max")
        strings = candidates[query["id"]]
        provided = [Chem.MolToInchiKey(Chem.MolFromSmiles(s)) for s in strings]
        cached, predictions, keys = {}, [], []
        for smiles in strings:
            entry = canonical_candidate(smiles, max_heavy_atoms=max_heavy_atoms)
            if entry is None:
                continue
            normalized, key = entry
            if normalized not in cached:
                cached[normalized] = forward(normalized, query)
            predictions.append(cached[normalized])
            keys.append(key)
        distances = expand_scores_to_full_candidates(keys, score_candidates(predictions, target),
                                                     provided)
        distances = np.nan_to_num(distances, nan=1.0)
        order = np.argsort(distances, kind="stable")
        rows.append({"id": query["id"], "candidates": [strings[i] for i in order],
                     "scores": (-distances[order]).tolist(), "status": "ok"})
    return rows
