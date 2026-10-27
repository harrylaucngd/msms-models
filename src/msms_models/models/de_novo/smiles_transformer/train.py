"""Train a formula-conditioned SMILES or SELFIES Transformer with MassSpecGym's settings.

The tokenizer is built first, as upstream builds it: byte-level BPE on the 4M MassSpecGym
molecules followed by the training molecules, or a SELFIES alphabet of the training molecules.
It is saved as ``tokenizer.json`` next to ``model.pt``.

    python -m msms_models.models.de_novo.smiles_transformer.train --tokenizer smiles-bpe \\
        --train data/msg/train.jsonl --validation data/msg/val.jsonl \\
        --molecules data/MassSpecGym_molecules_MCES2_disjoint_with_test_fold_4M.tsv \\
        --output checkpoints/smiles_transformer/smiles
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ....data.io import load_queries
from .inference import build, featurize
from .tokenizers import EOS, PAD, SOS, Tokenizer

# MassSpecGym's "v3" formula-challenge runs: 8 GPUs × 64 spectra, Adam at 3e-4, one validation
# per epoch, spectra tokenized up to m/z 1005.
DEFAULTS = {
    "d_model": 256, "num_decoder_layers": 4, "dropout": 0.1, "n_peaks": 60, "max_mz": 1005.0,
    "batch_size": 512, "lr": 3e-4, "max_epochs": 50, "val_check_interval": 1.0, "patience": 3,
    "seed": 0,
}
VARIANTS = {
    "smiles-bpe": {"nhead": 4, "num_encoder_layers": 3, "max_len": 200},
    "selfies": {"nhead": 8, "num_encoder_layers": 6, "max_len": 150},
}


def train(kind, train_queries, val_queries, output, *, molecules=(), device="cpu", **options):
    import torch
    from torch.utils.data import Dataset

    from ....massspecgym.training import fit

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    settings = DEFAULTS | VARIANTS[kind] | options
    train_smiles = [q["target_smiles"] for q in train_queries]
    tokenizer = Tokenizer.train(kind, train_smiles, molecules, settings["max_len"])
    tokenizer.save(output / "tokenizer.json")
    settings |= {"tokenizer": kind, "vocab_size": tokenizer.vocab_size(),
                 "pad_id": tokenizer.token_to_id(PAD), "sos_id": tokenizer.token_to_id(SOS),
                 "eos_id": tokenizer.token_to_id(EOS)}

    class Spectra(Dataset):
        def __init__(self, queries):
            self.spec, self.formula = (torch.from_numpy(a) for a in featurize(queries, settings))
            self.smiles = [q["target_smiles"] for q in queries]

        def __len__(self):
            return len(self.smiles)

        def __getitem__(self, i):
            return self.spec[i], self.formula[i], self.smiles[i]

    def collate(items):
        spec, formula, smiles = zip(*items)
        return torch.stack(spec), torch.stack(formula), list(smiles)

    def loss_fn(model, batch, device):
        spec, formula, smiles = batch
        tokens = torch.tensor(tokenizer.encode_batch(smiles), device=device)
        logits, targets = model(spec.to(device), formula.to(device), tokens)
        # Upstream's nn.CrossEntropyLoss() also scores padding tokens.
        return torch.nn.functional.cross_entropy(logits, targets)

    return fit(build(settings), Spectra(train_queries), Spectra(val_queries), loss_fn,
               settings=settings, output=output, collate=collate, device=device)


def read_molecules(path):
    import pandas as pd

    return pd.read_csv(path, sep="\t", usecols=["smiles"])["smiles"].tolist()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tokenizer", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--train", required=True, help="Training queries with target_smiles")
    parser.add_argument("--validation", required=True, help="Validation queries with target_smiles")
    parser.add_argument("--molecules", help="The 4M molecule TSV (smiles-bpe only)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    for key, value in (DEFAULTS | VARIANTS["smiles-bpe"]).items():
        parser.add_argument(f"--{key.replace('_', '-')}", type=type(value), default=None)
    args = vars(parser.parse_args(argv))
    options = {k: args[k] for k in DEFAULTS | VARIANTS["smiles-bpe"] if args[k] is not None}
    molecules = read_molecules(args["molecules"]) if args["tokenizer"] == "smiles-bpe" else ()
    best = train(args["tokenizer"], load_queries(args["train"]), load_queries(args["validation"]),
                 args["output"], molecules=molecules, device=args["device"], **options)
    print(json.dumps({"output": args["output"], "val_loss": best}))


if __name__ == "__main__":
    main()
