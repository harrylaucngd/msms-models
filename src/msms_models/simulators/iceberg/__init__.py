"""ICEBERG forward model: checkpoint loading and sparse spectrum prediction."""

from __future__ import annotations

import numpy as np

from ..retrieve import spectrum_settings
from ..vendor.spectral_scoring import bin_spectrum


def load_model(config):
    import torch

    from .vendor.dag_pred.gen_model import FragGNN
    from .vendor.dag_pred.inten_model import IntenGNN
    from .vendor.dag_pred.joint_model import JointModel

    paths = config.require_assets("generator_checkpoint", "intensity_checkpoint")
    payloads = [torch.load(paths[key], map_location="cpu", weights_only=False)
                for key in ("generator_checkpoint", "intensity_checkpoint")]
    # The adduct and instrument features follow the checkpoint: the released MSG weights
    # embed 11 adducts and 2 instruments, the CANOPUS weights 3 instruments, and the
    # Table 3 retrain 14 adducts and 4 instruments.
    state = payloads[0]["state_dict"]
    from .vendor import common

    common.use_vocabularies(
        state["adduct_embedder"].shape[0] if "adduct_embedder" in state else 11,
        state["instrument_embedder"].shape[0] if "instrument_embedder" in state else 3,
    )
    models = []
    for cls, payload in zip((FragGNN, IntenGNN), payloads):
        model = cls(**payload["hyper_parameters"])
        model.load_state_dict(payload["state_dict"])
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
    # Native ICEBERG first retains top-k unbinned fragment peaks, then sums bins.
    keep = np.argsort(output[:, 1])[::-1][:top]
    indices, values = bin_spectrum(output[keep], bins, upper, pool_fn="add")
    keep = values > 0
    return indices[keep], values[keep]

