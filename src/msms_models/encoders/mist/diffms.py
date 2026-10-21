"""MIST as released with DiffMS (Bohde et al., 2025): the "vDiffMS" encoder.

Reproduces the posterior path of MIST-MolForge (``mist-molforge-benchmark``), which
loads the DiffMS Zenodo checkpoint into DiffMS's ``SpectraEncoderGrowing`` and
featurizes DiffMS subformula trees. Spectra are encoded in query order in
consecutive batches of ``encoder_batch_size``, each right-padded to its longest
spectrum, as the upstream ``DataLoader(shuffle=False)`` does. Under
``attention_padding: additive-bool`` padding receives +1 instead of -inf (paper
Sec. 6.1.1), so posteriors depend on the batch size and composition; the default
``masked`` padding makes every posterior independent of its batch.

Only spectrum-side inputs are read: the subformula tree (precursor formula and
adduct, fragment formulas, intensities and ion types) of each query ID.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# The MassSpecGym checkpoint distributed with DiffMS (configs/spec2mol_benchmark_msg.yaml
# of MIST-MolForge); the CANOPUS checkpoint has a different width and is not wired here.
ARCHITECTURE = {
    "msg": dict(
        form_embedder="pos-cos", output_size=4096, hidden_size=512, spectra_dropout=0.1,
        top_layers=1, refine_layers=4, magma_modulo=2048, peak_attn_layers=2, num_heads=8,
        set_pooling="cls", pairwise_featurization=True, embed_instrument=False,
        inten_transform="float",
    ),
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


def build_model(dataset: str, attention_padding: str = "masked"):
    from .vendor.spectra_encoder import SpectraEncoderGrowing

    return SpectraEncoderGrowing(
        **ARCHITECTURE[dataset], additive_bool_mask=attention_padding == "additive-bool"
    )


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
    model = build_model(config.dataset, config.parameters.get("attention_padding", "masked"))
    # The Zenodo file is a bare SpectraEncoderGrowing state dict.
    model.load_state_dict(torch.load(paths["mist_checkpoint"], map_location="cpu"))
    device = torch.device(config.device)
    torch.set_num_threads(config.workers)
    model.to(device).eval()
    trees = load_trees(paths["diffms_features"], queries)
    # The instrument index is computed upstream but unused (embed_instrument=False),
    # so the unknown category is passed instead of reading benchmark labels.
    features = [featurize(tree, q["id"]) for tree, q in zip(trees, queries, strict=True)]
    return encode_batches(model, features, int(config.parameters.get("encoder_batch_size", 32)),
                          device)
