"""The benchmark-paired MIST architecture distributed with MS-GPT/FRIGID."""

from __future__ import annotations

import numpy as np


def load_features(path, queries):
    """Read annotated spectra by ID, without reading target structures or bits."""
    from ...data.annotated import iter_annotated_spectra

    columns = [
        "identifier", "formula", "precursor_mz", "adduct", "collision_energy", "peak_formula"
    ]
    wanted = {q["id"] for q in queries}
    found = {}
    for row in iter_annotated_spectra(path, columns=columns):
        identifier = row["identifier"]
        if identifier not in wanted:
            continue
        if identifier in found:
            raise ValueError(f"Duplicate annotated spectrum: {identifier}")
        found[identifier] = row
    result = []
    for query in queries:
        if query["id"] not in found:
            raise ValueError(f"Missing annotated spectrum: {query['id']}")
        row = found[query["id"]]
        if row["formula"] != query["formula"]:
            raise ValueError(f"Annotated spectrum formula differs for {query['id']}")
        result.append(row)
    return result


def encode(config, queries):
    import torch

    from .vendor.fingerprint import FingerprintGrowingModule
    from .vendor.nn_module import FormulaTransformer
    from .vendor.spectrum_utils import prepare_spectrum_batch

    paths = config.require_assets("mist_checkpoint", "legacy_features")
    records = load_features(paths["legacy_features"], queries)
    hidden = 640 if config.dataset == "msg" else 512
    padding = config.parameters.get("attention_padding", "masked")
    if padding not in {"masked", "additive-bool"}:
        raise ValueError("attention_padding must be masked or additive-bool")
    transformer = FormulaTransformer(
        hidden_size=hidden, peak_attn_layers=2, spectra_dropout=0.1,
        pairwise_featurization=True, num_heads=8, form_embedder="pos-cos",
        adduct_class_num=8, no_diffs=False, peak_type_set="full",
        collision_energy_encoder_type="sinusoidal", instrument_class_num=6,
        # additive-bool reproduces the historical padding bug (paper Sec. 6.1.1),
        # under which posteriors depend on the batch composition.
        additive_bool_mask=padding == "additive-bool",
    )
    grower = FingerprintGrowingModule(hidden_input_dim=hidden)
    payload = torch.load(paths["mist_checkpoint"], map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload)
    prefixes = ("spectra_encoder.0.", "spectra_encoder.2.")
    # The official MS-GPT posterior path bypasses this intermediate projection.
    unused = {"spectra_encoder.1.0.weight", "spectra_encoder.1.0.bias"}
    unexpected = {key for key in state if not key.startswith(prefixes)} - unused
    if unexpected:
        raise ValueError(f"Unknown legacy MIST tensors: {sorted(unexpected)}")
    ft = {key.removeprefix(prefixes[0]): value for key, value in state.items()
          if key.startswith(prefixes[0])}
    transformer.register_buffer("instrument_lookup", ft.pop("instrument_embedder"))
    result = transformer.load_state_dict(ft, strict=False)
    # Only deterministic/nonlearned runtime buffers may differ between releases.
    allowed = {"types_lookup", "adducts_lookup", "instrument_lookup"}
    if set(result.missing_keys) - allowed or result.unexpected_keys:
        raise ValueError(f"Incompatible legacy MIST transformer: {result}")
    grower.load_state_dict({
        key.removeprefix(prefixes[1]): value for key, value in state.items()
        if key.startswith(prefixes[1])
    }, strict=True)
    device = torch.device(config.device)
    torch.set_num_threads(config.workers)
    transformer.to(device).eval()
    grower.to(device).eval()
    values = []
    batch_size = int(config.parameters.get("encoder_batch_size", 32))
    if batch_size < 1:
        raise ValueError("encoder_batch_size must be positive")
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            tensors = prepare_spectrum_batch(
                [r["peak_formula"] for r in batch],
                [r["precursor_mz"] for r in batch],
                [r["formula"] for r in batch],
                [r["adduct"] for r in batch],
                [r["collision_energy"] for r in batch],
                device=str(device),
            )
            # Match the released MS-GPT encoder, which uses the unknown/zero
            # instrument category for these annotated benchmark inputs.
            tensors["batch_spec_instrument"] = torch.zeros(
                len(batch), dtype=torch.long, device=device
            )
            hidden_states = transformer(tensors)[tensors["batch_peak_type"] == 3]
            values.append(grower(hidden_states)[-1].float().cpu().numpy())
    return np.concatenate(values)
