"""Fine-tune the SMILES classifier on balanced query/decoy pairs from the training split.

    python -m msms_models.models.retrieval.chemberta_classifier.train \\
        --queries data/msg/train.jsonl \\
        --candidates data/MassSpecGym_retrieval_candidates_formula.json \\
        --output checkpoints/chemberta_classifier/inflated [--canonical]

Each training molecule contributes its own SMILES (label 1) and one decoy drawn
uniformly from its candidate list (label 0). No spectrum is used.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from ....data.candidates import load_candidates
from ....data.io import load_queries, write_json
from ....evaluation.chemistry import inchikey14
from .model import BASE_MODEL, load, text

DEFAULTS = {"epochs": 2, "batch_size": 32, "learning_rate": 5e-5, "max_length": 256, "seed": 0}


def training_pairs(queries, candidates, *, canonical: bool, seed: int):
    rng = random.Random(seed)
    pairs, seen = [], set()
    for query in queries:
        key = inchikey14(query["target_smiles"])
        if key in seen:  # one pair per molecule, not per spectrum
            continue
        seen.add(key)
        decoys = [c for c in candidates[query["id"]] if inchikey14(c) != key]
        if decoys:
            pairs.append((text(query["target_smiles"], canonical), 1))
            pairs.append((text(rng.choice(decoys), canonical), 0))
    return pairs


def train(queries, candidates, output, *, canonical=False, device="cpu", base_model=BASE_MODEL,
          **options):
    import torch

    settings = DEFAULTS | options
    torch.manual_seed(settings["seed"])
    pairs = training_pairs(queries, candidates, canonical=canonical, seed=settings["seed"])
    tokenizer, model = load(base_model, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"])
    rng = random.Random(settings["seed"])
    model.train()
    for epoch in range(settings["epochs"]):
        rng.shuffle(pairs)
        for start in range(0, len(pairs), settings["batch_size"]):
            batch = pairs[start:start + settings["batch_size"]]
            inputs = tokenizer(
                [s for s, _ in batch], padding=True, truncation=True,
                max_length=settings["max_length"], return_tensors="pt",
            ).to(device)
            labels = torch.tensor([y for _, y in batch], device=device)
            loss = model(**inputs, labels=labels).loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    output = Path(output)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    write_json(output / "training.json", {
        "base_model": base_model, "canonical_smiles": canonical, "pairs": len(pairs),
        "molecules": len(pairs) // 2, **settings,
    })
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--queries", required=True, help="Training queries with target_smiles")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--canonical", action="store_true", help="RDKit-canonicalize all SMILES")
    parser.add_argument("--device", default="cpu")
    for key, value in DEFAULTS.items():
        parser.add_argument(f"--{key.replace('_', '-')}", type=type(value), default=value)
    args = vars(parser.parse_args(argv))
    queries = load_queries(args.pop("queries"))
    candidates = load_candidates(args.pop("candidates"), queries)
    print(json.dumps({"output": str(train(queries, candidates, args.pop("output"), **args))}))


if __name__ == "__main__":
    main()
