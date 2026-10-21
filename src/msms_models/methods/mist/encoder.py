"""MIST spectrum encoding, shared by retrieval and fingerprint-conditioned methods."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ...datasets.download import sha256
from ...io import atomic_json
from ...posteriors import align_posterior


def select_feature_rows(names, queries):
    """Resolve exact IDs first, permitting only unambiguous acquisition aliases."""
    names = [str(name) for name in names]
    lookup = {name: i for i, name in enumerate(names)}
    if len(lookup) != len(names):
        raise ValueError("Duplicate spectrum IDs in MIST feature cache")
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
    return selected


def encode(config, queries):
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
    selected = select_feature_rows(names, queries)
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
    return save_posteriors(
        config, queries, values, paths["compact_cache"],
        source_checkpoint_sha256=checkpoint.get("source_checkpoint_sha256"),
    )


def save_posteriors(config, queries, values, features, *, source_checkpoint_sha256=None):
    checkpoint = config.require_assets("mist_checkpoint")["mist_checkpoint"]
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
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256(checkpoint),
            "source_checkpoint_sha256": source_checkpoint_sha256,
            "feature_cache": str(Path(features).resolve()),
        },
    )
    return values
