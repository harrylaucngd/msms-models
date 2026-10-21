"""MIST as released with DiffMS (Bohde et al., 2025): the "vDiffMS" encoder.

Reproduces the posterior path of MIST-MolForge (``mist-molforge-benchmark``), which
loads the DiffMS Zenodo checkpoint into DiffMS's ``SpectraEncoderGrowing`` and
featurizes DiffMS subformula trees. Spectra are encoded in query-manifest order in
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
PADDING = {"masked", "additive-bool"}


def load_trees(path, queries) -> list[dict]:
    """Subformula trees for the queries, from ``<id>.json`` files or an ID-keyed JSONL bundle."""
    path = Path(path)
    wanted = [q["id"] for q in queries]
    if path.is_dir():
        trees = {}
        for identifier in wanted:
            file = path / f"{identifier}.json"
            if not file.is_file():
                raise ValueError(f"Missing DiffMS subformula tree: {identifier}")
            trees[identifier] = json.loads(file.read_text())
    else:
        keep = set(wanted)
        trees = {}
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                identifier = row["id"]
                if identifier not in keep:
                    continue
                if identifier in trees:
                    raise ValueError(f"Duplicate DiffMS subformula tree: {identifier}")
                trees[identifier] = row["subformulae"]
        missing = [i for i in wanted if i not in trees]
        if missing:
            raise ValueError(f"Missing {len(missing)} DiffMS subformula trees: {missing[:5]}")
    result = []
    for query in queries:
        tree = trees[query["id"]]
        if tree.get("cand_form") != query["formula"]:
            raise ValueError(f"Subformula precursor formula differs for {query['id']}")
        if query.get("adduct") and tree.get("cand_ion") != query["adduct"]:
            raise ValueError(f"Subformula adduct differs for {query['id']}")
        result.append(tree)
    return result


def build_model(dataset: str, attention_padding: str = "masked"):
    from .vendor.diffms.spectra_encoder import SpectraEncoderGrowing

    if dataset not in ARCHITECTURE:
        raise ValueError(f"The DiffMS MIST encoder is configured only for {sorted(ARCHITECTURE)}")
    if attention_padding not in PADDING:
        raise ValueError("attention_padding must be masked or additive-bool")
    return SpectraEncoderGrowing(
        **ARCHITECTURE[dataset], additive_bool_mask=attention_padding == "additive-bool"
    )


def encode_batches(model, features, batch_size: int, device) -> np.ndarray:
    """Encode featurized spectra in order, ``batch_size`` consecutive spectra per pass."""
    import torch

    from .vendor.diffms.featurizers import collate_fn

    if batch_size < 1:
        raise ValueError("encoder_batch_size must be positive")
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

    from .vendor.diffms.featurizers import featurize

    paths = config.require_assets("mist_checkpoint", "diffms_features")
    padding = config.parameters.get("attention_padding", "masked")
    if padding == "additive-bool" and "encoder_batch_size" not in config.parameters:
        raise ValueError("additive-bool padding is batch-dependent; set encoder_batch_size")
    batch_size = int(config.parameters.get("encoder_batch_size", 32))
    model = build_model(config.dataset, padding)
    state = torch.load(paths["mist_checkpoint"], map_location="cpu", weights_only=False)
    state = state.get("state_dict", state)
    # The Zenodo file is a bare SpectraEncoderGrowing state dict, including the
    # fragment head and the pos-cos embedding tables; every tensor must match.
    model.load_state_dict(state, strict=True)
    device = torch.device(config.device)
    torch.set_num_threads(config.workers)
    model.to(device).eval()
    trees = load_trees(paths["diffms_features"], queries)
    # The instrument index is computed upstream but unused (embed_instrument=False),
    # so the unknown category is passed instead of reading benchmark labels.
    features = [featurize(tree, q["id"]) for tree, q in zip(trees, queries, strict=True)]
    return encode_batches(model, features, batch_size, device)
