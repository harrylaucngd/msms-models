"""Rank candidates by the classifier's probability that each SMILES is a query molecule."""

from __future__ import annotations

import json

from ....data.candidates import load_candidates
from .model import load, positive_probability, text


def predict(config, queries):
    import torch

    paths = config.require_assets("candidates", "classifier")
    training = json.loads((paths["classifier"] / "training.json").read_text())
    torch.set_num_threads(config.workers)
    tokenizer, model = load(str(paths["classifier"]), config.device)
    canonical = training["canonical_smiles"]
    candidates = load_candidates(paths["candidates"], queries)
    batch_size = int(config.parameters.get("batch_size", 256))
    # Test spectra share molecules and candidate pools: score each string once.
    unique = sorted({text(s, canonical) for q in queries for s in candidates[q["id"]]})
    probability = dict(zip(unique, positive_probability(
        tokenizer, model, unique, max_length=training["max_length"], batch_size=batch_size,
    )))
    rows = []
    for query in queries:
        strings = candidates[query["id"]]
        scores = [probability[text(s, canonical)] for s in strings]
        order = sorted(range(len(strings)), key=lambda i: -scores[i])
        rows.append({
            "id": query["id"], "candidates": [strings[i] for i in order],
            "scores": [scores[i] for i in order], "status": "ok",
        })
    return rows
