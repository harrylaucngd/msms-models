"""DeepSets fingerprint retrieval, a MassSpecGym baseline (Table 3: DeepSets, DeepSets + Fourier features).

Each spectrum's peaks, led by the precursor (``SpecTokenizer``), are embedded one by one,
summed and mapped to a 4,096-bit Morgan fingerprint; candidates are ranked by cosine
similarity. Train with the settings of MassSpecGym's formula-challenge run:

    python -m msms_models.models.retrieval.deepsets --train data/msg/train.jsonl \\
        --validation data/msg/val.jsonl --output checkpoints/deepsets/fourier
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from ...data.io import load_queries
from ...massspecgym.retrieval import fit_fingerprints, predict_fingerprints
from ...massspecgym.transforms import tokenize_spectrum

# MassSpecGym's formula-challenge run: 8 GPUs × 128 spectra, Adam at 1e-3, two validations per
# epoch. The plain DeepSets row predates Fourier features and used m/z up to 1000.
DEFAULTS = {
    "fourier_features": True, "hidden_channels": 128, "num_layers_per_mlp": 4, "dropout": 0.1,
    "n_peaks": 60, "max_mz": 1005.0, "batch_size": 1024, "lr": 1e-3, "max_epochs": 50,
    "val_check_interval": 0.5, "patience": 3, "seed": 0,
}


def build(settings):
    from ...massspecgym.networks import DeepSets

    return DeepSets(settings["hidden_channels"], 4096, settings["num_layers_per_mlp"],
                    settings["dropout"], settings["fourier_features"])


def featurize(queries, settings):
    return np.stack([
        tokenize_spectrum(q, n_peaks=settings["n_peaks"], max_mz=settings["max_mz"]) for q in queries
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
        if isinstance(value, bool):
            parser.add_argument(f"--{key.replace('_', '-')}", action=argparse.BooleanOptionalAction,
                                default=value)
        else:
            parser.add_argument(f"--{key.replace('_', '-')}", type=type(value), default=value)
    args = vars(parser.parse_args(argv))
    settings = {key: args[key] for key in DEFAULTS}
    best = fit_fingerprints(build, featurize, load_queries(args["train"]),
                            load_queries(args["validation"]), settings, args["output"],
                            device=args["device"])
    print(json.dumps({"output": args["output"], "val_loss": best}))


if __name__ == "__main__":
    main()
