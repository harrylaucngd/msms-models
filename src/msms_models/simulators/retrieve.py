"""Candidate retrieval by forward spectral prediction and native entropy distance."""

from __future__ import annotations

import numpy as np
from rdkit import Chem

from ..data.candidates import load_candidates
from ..data.io import atomic_json
from .vendor.spectral_scoring import (
    _entropy_sparse,
    bin_spectrum,
    expand_scores_to_full_candidates,
)


class UnsupportedCandidate(ValueError):
    """A candidate is outside the declared forward-model chemical domain."""


def canonical_candidate(smiles, *, max_heavy_atoms=10000):
    # Common ms-pred chemical domain, independent of a model runtime.
    elements = {"C", "N", "P", "O", "S", "Si", "I", "H", "Cl", "F", "Br", "B", "Se", "Fe", "Co", "As", "Na", "K"}

    mol = Chem.MolFromSmiles(smiles)
    if mol is None or not mol.GetNumAtoms():
        raise UnsupportedCandidate("invalid_structure")
    if any(a.GetSymbol() not in elements for a in mol.GetAtoms()):
        raise UnsupportedCandidate("unsupported_element")
    mol = Chem.RemoveHs(mol)
    if mol.GetNumHeavyAtoms() > max_heavy_atoms:
        raise UnsupportedCandidate("heavy_atom_limit")
    # This is predict_smis_joint.prepare_entry's normalization, not tautomerization.
    return Chem.MolToSmiles(mol), Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))


def spectrum_settings(config):
    bins = int(config.parameters.get("num_bins", 150000))
    upper = float(config.parameters.get("upper_limit", 1500))
    top = int(config.parameters.get("sparse_k", 100))
    if top < 1 or top > bins or bins < 2 or upper <= 0:
        raise ValueError("Invalid spectral binning or sparse_k")
    return bins, upper, top


def validate_metadata(query):
    required = ("spectrum", "collision_energy", "adduct", "instrument", "precursor_mz")
    missing = [name for name in required if name not in query]
    if missing:
        raise ValueError(f"Query {query['id']} requires {missing}; collision energy is in eV")
    if query["collision_energy"] < 0 or not np.isfinite(query["collision_energy"]):
        raise ValueError("Collision energy must be nonnegative and finite")
    if query["adduct"] not in {"[M+H]+", "[M+Na]+"}:
        raise ValueError(f"Unsupported forward-model adduct: {query['adduct']}")
    if query["instrument"] not in {"Orbitrap", "QTOF"}:
        raise ValueError(f"Unsupported forward-model instrument: {query['instrument']}")


def score_candidates(predictions, target):
    """Match native single-energy aggregation, including its normalization epsilon."""
    distances = _entropy_sparse([{0: p} for p in predictions], target, 0, sqrt_inten=True)
    n = int(np.count_nonzero(target[1] > 0))
    weight = int(n >= 5) * 3 + int(n >= 1)
    return distances * (np.float64(weight) / (weight + 1e-9))


def retrieve(config, queries, forward, *, prediction_pooling):
    """Rank supplied candidates using a method-owned spectral predictor."""

    paths = config.require_assets("candidates")
    candidates = load_candidates(paths["candidates"], queries)
    # Explicit input contract: the paper's assigned fragment spectra already have
    # transformed intensities. Raw peak intensities must opt into one sqrt here.
    intensity_space = config.parameters.get("query_intensity_space")
    if intensity_space not in {"native", "raw"}:
        raise ValueError(
            "Set query_intensity_space to native (prepared spectra) or raw (sqrt once)"
        )
    for query in queries:
        validate_metadata(query)
    bins, upper, top = spectrum_settings(config)
    rows = []
    unsupported = {}
    for query in queries:
        raw = np.column_stack([query["spectrum"]["mz"], query["spectrum"]["intensity"]])
        if intensity_space == "raw":
            raw[:, 1] = np.sqrt(raw[:, 1])
        target = bin_spectrum(raw, bins, upper, pool_fn="max")
        if not np.any(target[1] > 0):
            raise ValueError(f"Query {query['id']} has no positive peaks within the bin range")
        strings = candidates[query["id"]]
        molecules = [Chem.MolFromSmiles(s) for s in strings]
        provided = [Chem.MolToInchiKey(m) if m is not None else "" for m in molecules]
        if any(not key for key in provided):
            raise ValueError(f"Invalid candidate identity in query {query['id']}")
        cached, predictions, keys = {}, [], []
        for smi in strings:
            try:
                normalized, key = canonical_candidate(
                    smi, max_heavy_atoms=int(config.parameters.get("max_heavy_atoms", 10000))
                )
            except UnsupportedCandidate as error:
                unsupported[str(error)] = unsupported.get(str(error), 0) + 1
                continue
            if normalized not in cached:
                cached[normalized] = forward(normalized, query)
            predictions.append(cached[normalized])
            keys.append(key)
        distances = score_candidates(predictions, target)
        full = expand_scores_to_full_candidates(keys, distances, provided)
        full = np.nan_to_num(full, nan=1.0)
        if not np.isfinite(full).all():
            raise ValueError("Forward model produced non-finite spectral distances")
        order = np.argsort(full, kind="stable")
        rows.append(
            {
                "id": query["id"],
                "candidates": [strings[i] for i in order],
                "scores": (-full[order]).tolist(),
                "status": "ok",
            }
        )
    atomic_json(
        config.output / "spectral_protocol.json",
        {
            "metric": "native entropy distance",
            "num_bins": bins,
            "upper_limit": upper,
            "sparse_k": top,
            "query_intensity_space": intensity_space,
            "query_pooling": "max",
            "prediction_pooling": prediction_pooling,
            "ties": "stable",
            "missing_prediction_distance": 1.0,
            "unsupported_candidate_counts": unsupported,
            "collision_energy_unit": "eV",
            "checkpoint_loading": "strict",
        },
    )
    return rows
