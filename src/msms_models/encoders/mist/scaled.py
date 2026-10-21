"""ScaledMIST spectrum encoding (coleygroup/MIST), e.g. the MIST 4M-aug checkpoint."""

from __future__ import annotations

import numpy as np


def select_feature_rows(names, queries):
    """Feature-cache row of each query: its exact ID, else its ``<id>_collision …`` entry."""
    names = [str(name) for name in names]
    lookup = {name: i for i, name in enumerate(names)}
    for i, name in enumerate(names):
        lookup.setdefault(name.split("_collision ", 1)[0], i)
    return [lookup[q["id"]] for q in queries]


def encode(config, queries):
    import torch
    from mist.data.input_signal import SpectrumOnlyDataset, collate_spectrum_only
    from mist.models.scaled_mist import ScaledMIST
    from torch.utils.data import DataLoader, Subset

    paths = config.require_assets("compact_cache", "mist_checkpoint")
    torch.set_num_threads(config.workers)
    dataset = SpectrumOnlyDataset(paths["compact_cache"])
    checkpoint = torch.load(paths["mist_checkpoint"], map_location="cpu", weights_only=False)
    model = ScaledMIST(**checkpoint["config"]["model"])
    model.load_state_dict(checkpoint["ema_model"])
    device = torch.device(config.device)
    model.to(device).eval()
    loader = DataLoader(
        Subset(dataset, select_feature_rows(dataset.names().astype(str), queries)),
        batch_size=int(config.parameters.get("encoder_batch_size", 64)),
        shuffle=False, collate_fn=collate_spectrum_only,
    )
    predictions = []
    with torch.inference_mode():
        for batch in loader:
            values, _ = model({k: v.to(device) for k, v in batch.items()})
            predictions.append(values.float().cpu().numpy())
    return np.concatenate(predictions)
