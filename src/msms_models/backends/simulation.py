"""Forward spectral prediction and native entropy candidate retrieval."""

from __future__ import annotations

import numpy as np
from rdkit import Chem

from .._vendor.spectral_scoring import (
    _entropy_sparse,
    bin_spectrum,
    expand_scores_to_full_candidates,
)
from ..io import atomic_json
from ..retrieval.fingerprints import load_candidates


class UnsupportedCandidate(ValueError):
    """A candidate is outside the declared forward-model chemical domain."""


def canonical_candidate(smiles, *, max_heavy_atoms=10000):
    from msms_models._vendor.iceberg import common
    from msms_models._vendor.iceberg.common import chem_utils

    mol = Chem.MolFromSmiles(smiles)
    if mol is None or not mol.GetNumAtoms():
        raise UnsupportedCandidate("invalid_structure")
    if any(a.GetAtomicNum() not in chem_utils.VALID_ATOM_NUM for a in mol.GetAtoms()):
        raise UnsupportedCandidate("unsupported_element")
    mol = Chem.RemoveHs(mol)
    if mol.GetNumHeavyAtoms() > max_heavy_atoms:
        raise UnsupportedCandidate("heavy_atom_limit")
    # This is predict_smis_joint.prepare_entry's normalization, not tautomerization.
    return Chem.MolToSmiles(mol), common.inchikey_from_smiles(smiles)


def load_forward(config):
    import torch

    if config.model == "glacier":
        from mist.glacier.predict import load_glacier

        return load_glacier(config.require_assets("checkpoint")["checkpoint"], config.device)
    import importlib

    family = config.parameters.get("iceberg_module", "dag_pred")
    if family != "dag_pred":
        raise ValueError("This release supports the dag_pred ICEBERG checkpoint architecture")
    FragGNN = importlib.import_module(f"msms_models._vendor.iceberg.{family}.gen_model").FragGNN
    IntenGNN = importlib.import_module(f"msms_models._vendor.iceberg.{family}.inten_model").IntenGNN
    JointModel = importlib.import_module(
        f"msms_models._vendor.iceberg.{family}.joint_model"
    ).JointModel
    paths = config.require_assets("generator_checkpoint", "intensity_checkpoint")
    models = []
    for cls, key in [(FragGNN, "generator_checkpoint"), (IntenGNN, "intensity_checkpoint")]:
        payload = torch.load(paths[key], map_location="cpu", weights_only=False)
        if family == "dag_pred" and "instrument_embedder" in payload["state_dict"]:
            from msms_models._vendor.iceberg import common

            count = payload["state_dict"]["instrument_embedder"].shape[0]
            if count not in {2, 3}:
                raise ValueError("Unsupported legacy ICEBERG instrument vocabulary")
            vocabulary = {"Orbitrap": 0, "QTOF": 1}
            if count == 3:
                vocabulary["Unknown"] = 2
            # Released legacy MSG and CANOPUS weights have respectively two and
            # three instrument features. Keep their original feature positions.
            common.instrument2onehot_pos.clear()
            common.instrument2onehot_pos.update(vocabulary)
        model = cls(**payload["hyper_parameters"])
        model.load_state_dict(payload["state_dict"], strict=True)
        models.append(model)
    return JointModel(*models).to(config.device).eval().requires_grad_(False)


def forward_spectrum(model, config, smiles, query):
    """Return the native sparse stored spectrum (model intensities, no extra sqrt)."""
    import torch

    params = config.parameters
    bins = int(params.get("num_bins", 150000))
    upper = float(params.get("upper_limit", 1500))
    top = int(params.get("sparse_k", 100))
    if top < 1 or top > bins or bins < 2 or upper <= 0:
        raise ValueError("Invalid spectral binning or sparse_k")
    with torch.inference_mode():
        if config.model == "glacier":
            output = model.predict_mol(
                smiles,
                query["collision_energy"],
                query["adduct"],
                config.device,
                query["instrument"],
            )["spec"]
            if not torch.isfinite(output).all():
                raise ValueError("GLACIER produced a non-finite spectrum")
            # Native GLACIER keeps top-k *binned* peaks, using float32 torch arithmetic.
            index = torch.floor(output[:, 0] * ((bins - 1) / upper)).long() + 1
            keep = (output[:, 1] > 0) & (index >= 0) & (index < bins)
            dense = torch.zeros(bins, dtype=output.dtype, device=output.device)
            dense.scatter_reduce_(0, index[keep], output[keep, 1], reduce="amax", include_self=True)
            values, indices = torch.topk(dense, k=top)
            indices, values = indices.cpu().numpy(), values.cpu().numpy()
            keep = values > 0
            indices, values = indices[keep], values[keep]
            order = np.argsort(indices, kind="stable")
            return indices[order].astype(np.uint32), values[order].astype(np.float32)
        output = model.predict_mol(
            smiles,
            query["collision_energy"],
            query["precursor_mz"],
            query["adduct"],
            float(params.get("fragment_threshold", 0.0)),
            config.device,
            int(params.get("max_nodes", 100)),
            instrument=query["instrument"],
            canonical_root_smi=True,
            binned_out=False,
            adduct_shift=True,
        )["spec"]
        output = (
            output.detach().cpu().numpy()
            if isinstance(output, torch.Tensor)
            else np.asarray(output)
        )
    if not np.isfinite(output).all():
        raise ValueError("ICEBERG produced a non-finite spectrum")
    # Native ICEBERG first retains top-k unbinned fragment peaks, then sums bins.
    keep = np.argsort(output[:, 1])[::-1][:top]
    indices, values = bin_spectrum(output[keep], bins, upper, pool_fn="add")
    keep = values > 0
    return indices[keep], values[keep]


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


def predict(config, queries):
    import torch

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
    torch.set_num_threads(config.workers)
    torch.manual_seed(config.seed)
    model = load_forward(config)
    bins = int(config.parameters.get("num_bins", 150000))
    upper = float(config.parameters.get("upper_limit", 1500))
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
                cached[normalized] = forward_spectrum(model, config, normalized, query)
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
            "sparse_k": int(config.parameters.get("sparse_k", 100)),
            "query_intensity_space": intensity_space,
            "query_pooling": "max",
            "prediction_pooling": "max" if config.model == "glacier" else "add",
            "ties": "stable",
            "missing_prediction_distance": 1.0,
            "unsupported_candidate_counts": unsupported,
            "collision_energy_unit": "eV",
            "checkpoint_loading": "strict",
        },
    )
    return rows
