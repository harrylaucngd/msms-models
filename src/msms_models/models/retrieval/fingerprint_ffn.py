"""Fingerprint FFN retrieval, a MassSpecGym baseline (Table 3: Fingerprint FFN).

A spectrum binned at 1 Da up to m/z 1005 (``SpecBinner``) is mapped by an MLP to a 4,096-bit
Morgan fingerprint; candidates are ranked by cosine similarity. Train with the settings of
MassSpecGym's formula-challenge run:

    python -m msms_models.models.retrieval.fingerprint_ffn --train data/msg/train.jsonl \\
        --validation data/msg/val.jsonl --output checkpoints/fingerprint_ffn
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from ...data.io import load_queries
from ...massspecgym.retrieval import fit_fingerprints, predict_fingerprints
from ...massspecgym.transforms import bin_spectrum

# MassSpecGym's formula-challenge run: 8 GPUs × 64 spectra, Adam at 1e-3, two validations
# per epoch.
DEFAULTS = {
    "max_mz": 1005.0, "bin_width": 1.0, "hidden_channels": 128, "num_layers": 7, "dropout": 0.1,
    "batch_size": 512, "lr": 1e-3, "max_epochs": 50, "val_check_interval": 0.5, "patience": 3,
    "seed": 0,
}


def build(settings):
    from ...massspecgym.networks import FingerprintFFN

    return FingerprintFFN(int(settings["max_mz"] / settings["bin_width"]),
                          settings["hidden_channels"], 4096, settings["num_layers"],
                          settings["dropout"])


def featurize(queries, settings):
    return np.stack([
        bin_spectrum(q, max_mz=settings["max_mz"], bin_width=settings["bin_width"]) for q in queries
    ]).astype(np.float32)


def predict(config, queries):
    return predict_fingerprints(config, queries, build, featurize)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", required=True, help="Training queries with target_smiles")
    parser.add_argument("--validation", required=True, help="Validation queries with target_smiles")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    for key, value in DEFAULTS.items():
        parser.add_argument(f"--{key.replace('_', '-')}", type=type(value), default=value)
    args = vars(parser.parse_args(argv))
    settings = {key: args[key] for key in DEFAULTS}
    best = fit_fingerprints(build, featurize, load_queries(args["train"]),
                            load_queries(args["validation"]), settings, args["output"],
                            device=args["device"])
    print(json.dumps({"output": args["output"], "val_loss": best}))


if __name__ == "__main__":
    main()
