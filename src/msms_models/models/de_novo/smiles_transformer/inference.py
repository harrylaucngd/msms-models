"""Sample molecules from a trained SMILES or SELFIES Transformer."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ....massspecgym.transforms import formula_vector, tokenize_spectrum


def featurize(queries, settings):
    """Tokenized spectra and formula vectors, as float32 arrays."""
    spec = np.stack([
        tokenize_spectrum(q, n_peaks=settings["n_peaks"], max_mz=settings["max_mz"]) for q in queries
    ]).astype(np.float32)
    formula = np.stack([formula_vector(q["formula"]) for q in queries]).astype(np.float32)
    return spec, formula


def build(settings):
    from .model import SmilesTransformer

    return SmilesTransformer(
        settings["vocab_size"], settings["pad_id"], settings["d_model"], settings["nhead"],
        settings["num_encoder_layers"], settings["num_decoder_layers"], settings["dropout"],
    )


def predict(config, queries):
    import torch

    from ....massspecgym.training import load
    from .tokenizers import Tokenizer

    checkpoint = config.require_assets("checkpoint")["checkpoint"]
    model, settings = load(checkpoint, build, config.device)
    tokenizer = Tokenizer.load(Path(checkpoint).parent / "tokenizer.json", settings["tokenizer"])
    k = int(config.parameters.get("k_predictions", 10))
    # Upstream decodes greedily when a single prediction is asked for.
    temperature = float(config.parameters.get("temperature", 1.0)) if k > 1 else None
    batch_size = int(config.parameters.get("batch_size", 64))
    torch.manual_seed(config.seed)
    spec, formula = featurize(queries, settings)
    rows = []
    for start in range(0, len(queries), batch_size):
        x = torch.from_numpy(spec[start:start + batch_size]).to(config.device)
        f = torch.from_numpy(formula[start:start + batch_size]).to(config.device)
        samples = [
            tokenizer.decode_batch(model.decode(
                x, f, sos_id=settings["sos_id"], eos_id=settings["eos_id"],
                max_len=settings["max_len"], temperature=temperature,
            ).tolist())
            for _ in range(k)
        ]
        for i, query in enumerate(queries[start:start + batch_size]):
            rows.append({"id": query["id"], "candidates": [s[i] for s in samples], "status": "ok"})
    return rows
