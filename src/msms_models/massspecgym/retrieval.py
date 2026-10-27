"""Training and prediction shared by the fingerprint-predicting MassSpecGym retrieval baselines."""

from __future__ import annotations

import numpy as np

from .transforms import morgan_fingerprint


def cosine_loss(model, batch, device):
    """``CosSimLoss``: one minus the mean cosine similarity of true and predicted fingerprints."""
    from torch.nn import functional as F

    x, fingerprints = (t.to(device) for t in batch)
    return 1 - F.cosine_similarity(fingerprints, model(x)).mean()


def fit_fingerprints(build, featurize, train_queries, val_queries, settings, output, *,
                     device="cpu"):
    """Train ``build(settings)`` to predict each target's 4,096-bit Morgan fingerprint."""
    import torch
    from torch.utils.data import TensorDataset

    from .training import fit

    def dataset(queries):
        x = torch.from_numpy(featurize(queries, settings))
        y = torch.from_numpy(np.stack([morgan_fingerprint(q["target_smiles"]) for q in queries]))
        return TensorDataset(x, y.float())

    return fit(build(settings), dataset(train_queries), dataset(val_queries), cosine_loss,
               settings=settings, output=output, device=device)


def predict_fingerprints(config, queries, build, featurize):
    """Rank each query's candidates by cosine similarity to the predicted fingerprint."""
    import torch

    from ..data.candidates import load_candidates
    from ..models.retrieval.fingerprint import rank_fingerprints
    from .training import load

    paths = config.require_assets("checkpoint", "candidates")
    model, settings = load(paths["checkpoint"], build, config.device)
    batch_size = int(config.parameters.get("batch_size", 256))
    x = featurize(queries, settings)
    with torch.no_grad():
        fingerprints = np.concatenate([
            model(torch.from_numpy(x[i:i + batch_size]).to(config.device)).cpu().numpy()
            for i in range(0, len(x), batch_size)
        ])
    return rank_fingerprints(queries, fingerprints, load_candidates(paths["candidates"], queries),
                             similarity="cosine")
