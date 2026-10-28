"""MIST in the MistNet form of its DiffMS and MassSpecGym releases.

``mist_architecture: diffms`` is the checkpoint distributed with DiffMS (Bohde et al., 2025),
the "vDiffMS" encoder; ``massspecgym`` is MIST vMassSpecGym (Zenodo 11580401). Both read
the subformula tree of each query ID (precursor formula and adduct, fragment formulas,
intensities and ion types). Spectra are encoded in query order in consecutive batches of
``encoder_batch_size``, each right-padded to its longest spectrum. Under
``attention_padding: additive-bool`` padding receives +1 instead of -inf (paper
Sec. 6.1.1), so posteriors depend on the batch; ``masked`` makes them independent of it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_COMMON = dict(form_embedder="pos-cos", output_size=4096, hidden_size=512, top_layers=1,
               num_heads=8, set_pooling="cls", pairwise_featurization=True,
               embed_instrument=False, inten_transform="float")
ARCHITECTURE = {
    "diffms": _COMMON | dict(spectra_dropout=0.1, refine_layers=4, magma_modulo=2048,
                             peak_attn_layers=2),
    "massspecgym": _COMMON | dict(spectra_dropout=0.3, refine_layers=5, magma_modulo=512,
                                  peak_attn_layers=3),
}


def load_trees(path, queries) -> list[dict]:
    """Subformula trees for the queries, from ``<id>.json`` files or an ID-keyed JSONL bundle."""
    path = Path(path)
    if path.is_dir():
        return [json.loads((path / f"{q['id']}.json").read_text()) for q in queries]
    wanted = {q["id"] for q in queries}
    trees = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["id"] in wanted:
                trees[row["id"]] = row["subformulae"]
    return [trees[q["id"]] for q in queries]


def build_model(architecture: str = "diffms", attention_padding: str = "masked"):
    from .vendor.spectra_encoder import SpectraEncoderGrowing

    return SpectraEncoderGrowing(
        **ARCHITECTURE[architecture], additive_bool_mask=attention_padding == "additive-bool"
    )


def load_state(path):
    """A bare encoder state dict (DiffMS) or a Lightning checkpoint of MIST (MassSpecGym)."""
    import torch

    state = torch.load(path, map_location="cpu", weights_only=False)
    if "state_dict" in state:
        # The Lightning module holds the encoder under the same ``spectra_encoder`` name.
        state = {k: v for k, v in state["state_dict"].items() if k.startswith("spectra_encoder.")}
    return state


def encode_batches(model, features, batch_size: int, device) -> np.ndarray:
    """Encode featurized spectra in order, ``batch_size`` consecutive spectra per pass."""
    import torch

    from .vendor.featurizers import collate_fn

    values = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            batch = collate_fn(features[start:start + batch_size])
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            probabilities, _ = model(batch)
            values.append(probabilities.float().cpu().numpy())
    return np.concatenate(values)


def encode(config, queries) -> np.ndarray:
    import torch

    from .vendor.featurizers import featurize

    paths = config.require_assets("mist_checkpoint", "diffms_features")
    p = config.parameters
    model = build_model(p.get("mist_architecture", "diffms"), p.get("attention_padding", "masked"))
    model.load_state_dict(load_state(paths["mist_checkpoint"]))
    device = torch.device(config.device)
    torch.set_num_threads(config.workers)
    model.to(device).eval()
    trees = load_trees(paths["diffms_features"], queries)
    features = [featurize(tree, q["id"]) for tree, q in zip(trees, queries, strict=True)]
    return encode_batches(model, features, int(p.get("encoder_batch_size", 32)), device)
