"""ICEBERG checkpoint loading, sparse spectral prediction and candidate retrieval."""

from __future__ import annotations

import numpy as np

from ..._vendor.spectral_scoring import bin_spectrum
from ...simulation import retrieve, spectrum_settings


def load_model(config):
    import torch

    from .vendor.dag_pred.gen_model import FragGNN
    from .vendor.dag_pred.inten_model import IntenGNN
    from .vendor.dag_pred.joint_model import JointModel

    family = config.parameters.get("iceberg_module", "dag_pred")
    if family != "dag_pred":
        raise ValueError("This release supports the dag_pred ICEBERG checkpoint architecture")
    paths = config.require_assets("generator_checkpoint", "intensity_checkpoint")
    models = []
    for cls, key in [(FragGNN, "generator_checkpoint"), (IntenGNN, "intensity_checkpoint")]:
        payload = torch.load(paths[key], map_location="cpu", weights_only=False)
        if family == "dag_pred" and "instrument_embedder" in payload["state_dict"]:
            from msms_models.methods.iceberg.vendor import common

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


def predict_spectrum(model, config, smiles, query):
    """Retain top fragment peaks before summing their binned intensities."""
    import torch

    params = config.parameters
    bins, upper, top = spectrum_settings(config)
    with torch.inference_mode():
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


def predict(config, queries):
    import torch

    torch.set_num_threads(config.workers)
    torch.manual_seed(config.seed)
    model = load_model(config)
    return retrieve(
        config, queries,
        lambda smiles, query: predict_spectrum(model, config, smiles, query),
        prediction_pooling="add",
    )
