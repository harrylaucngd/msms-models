"""Spectrum-only MIST encoding shared by fingerprint-conditioned decoders."""

from __future__ import annotations

import json

import numpy as np

from .config import RunConfig
from .io import atomic_json


def align_posterior(values, ids, queries):
    values = np.asarray(values)
    ids = [str(i) for i in ids]
    if values.shape != (len(ids), 4096):
        raise ValueError("Posteriors must be a row-aligned N x 4096 matrix")
    if len(ids) != len(set(ids)):
        raise ValueError("Posterior ids must be unique")
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1.000001).any():
        raise ValueError("Fingerprint probabilities must be finite and in [0, 1]")
    index = {identifier: i for i, identifier in enumerate(ids)}
    missing = [q["id"] for q in queries if q["id"] not in index]
    if missing:
        raise ValueError(f"Posterior cache is missing {len(missing)} queries: {missing[:5]}")
    return np.asarray(values[[index[q["id"]] for q in queries]], dtype=np.float32)


def get_posteriors(config: RunConfig, queries: list[dict]) -> np.ndarray:
    if "posterior" in config.assets:
        paths = config.require_assets("posterior", "posterior_ids", "posterior_manifest")
        manifest = json.loads(paths["posterior_manifest"].read_text())
        if manifest.get("dataset") != config.dataset:
            raise ValueError("Posterior manifest dataset differs from the requested dataset")
        if manifest.get("fingerprint") != "morgan-r2-4096-achiral":
            raise ValueError("Unsupported posterior fingerprint definition")
        return align_posterior(
            np.load(paths["posterior"], mmap_mode="r"),
            np.load(paths["posterior_ids"], allow_pickle=False),
            queries,
        )
    paths = config.require_assets("compact_cache", "mist_checkpoint")
    try:
        import torch
        from mist.data.input_signal import SpectrumOnlyDataset, collate_spectrum_only
        from mist.models.scaled_mist import ScaledMIST
        from torch.utils.data import DataLoader, Subset
    except ImportError as error:
        raise RuntimeError(
            "MIST encoding requires the public coleygroup/MIST package in this backend environment"
        ) from error
    torch.set_num_threads(config.workers)
    dataset = SpectrumOnlyDataset(paths["compact_cache"])
    expected = "msg_all" if config.dataset == "msg" else "canopus_all"
    if dataset.manifest.get("dataset_name") != expected:
        raise ValueError(f"Expected {expected} MIST features")
    names = dataset.names().astype(str)
    # Released compact caches may encode acquisition labels in their row names.
    # Exact matches take precedence; normalized aliases must remain unambiguous.
    lookup = {name: i for i, name in enumerate(names)}
    aliases = {}
    for i, name in enumerate(names):
        alias = name.split("_collision ", 1)[0]
        aliases.setdefault(alias, []).append(i)
    selected = []
    for query in queries:
        identifier = query["id"]
        if identifier in lookup:
            selected.append(lookup[identifier])
        elif len(aliases.get(identifier, [])) == 1:
            selected.append(aliases[identifier][0])
        else:
            raise ValueError(f"Missing or ambiguous MIST feature row for {identifier}")
    checkpoint = torch.load(paths["mist_checkpoint"], map_location="cpu", weights_only=False)
    if checkpoint["config"].get("data", {}).get("dataset_name") != expected:
        raise ValueError("MIST checkpoint dataset differs from the requested dataset")
    model = ScaledMIST(**checkpoint["config"]["model"])
    model.load_state_dict(checkpoint["ema_model"], strict=True)
    device = torch.device(config.device)
    model.to(device).eval()
    loader = DataLoader(
        Subset(dataset, selected),
        batch_size=int(config.parameters.get("encoder_batch_size", 64)),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_spectrum_only,
    )
    predictions = []
    with torch.inference_mode():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            values, _ = model(batch)
            predictions.append(values.float().cpu().numpy())
    values = np.concatenate(predictions)
    ids = np.array([q["id"] for q in queries])
    values = align_posterior(values, ids, queries)
    output = config.output / "posterior"
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "probabilities.npy", values)
    np.save(output / "query_ids.npy", ids)
    atomic_json(
        output / "manifest.json",
        {
            "schema_version": 1,
            "dataset": config.dataset,
            "rows": len(ids),
            "fingerprint": "morgan-r2-4096-achiral",
            "target_fingerprints_read": False,
            "checkpoint": str(paths["mist_checkpoint"].resolve()),
            "feature_cache": str(paths["compact_cache"].resolve()),
        },
    )
    return values
