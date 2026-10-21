"""Fingerprint cache validation and ID alignment for conditioned methods."""

from __future__ import annotations

import json

import numpy as np

from .config import RunConfig


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
    if "legacy_features" in config.assets:
        from .methods.mist.encoder import save_posteriors
        from .methods.mist.legacy import encode

        values = encode(config, queries)
        return save_posteriors(config, queries, values, config.assets["legacy_features"])
    from .methods.mist.encoder import encode

    return encode(config, queries)
