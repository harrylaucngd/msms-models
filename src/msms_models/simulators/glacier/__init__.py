"""GLACIER forward model: checkpoint loading and sparse spectrum prediction."""

from __future__ import annotations

import numpy as np

from ..retrieve import spectrum_settings


def load_model(config):
    import sys

    from .. import scatter

    # ms-pred imports torch_scatter for these reductions; see ..scatter for why they come
    # from PyTorch here.
    sys.modules.setdefault("torch_scatter", scatter)
    from mist.glacier.predict import load_glacier

    return load_glacier(config.require_assets("checkpoint")["checkpoint"], config.device)


def predict_spectrum(model, config, smiles, query):
    """Max-pool peaks before taking the highest-intensity bins, as in GLACIER."""
    import torch

    bins, upper, top = spectrum_settings(config)
    with torch.inference_mode():
        output = model.predict_mol(
            smiles,
            query["collision_energy"],
            query["adduct"],
            config.device,
            query["instrument"],
        )["spec"]
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

