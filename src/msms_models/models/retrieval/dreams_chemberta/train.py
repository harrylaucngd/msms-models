"""Train the SpecBridge adapter and mapper on annotated spectra (upstream `train_real`).

    python -m msms_models.models.retrieval.dreams_chemberta.train --config train.yaml

The defaults are the settings stored in both Table 3 checkpoints. Only the code
path those settings exercise is retained: frozen ChemBERTa, a non-Gaussian
mapper, no early stopping, no mixed precision and in-process data loading.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import yaml

from ....data.io import load_queries

CHEMBERTA = "Derify/ChemBERTa_augmented_pubchem_13m"
DEFAULTS = {
    "train": None,  # annotated spectra with target_smiles (JSONL, TSV, ...)
    "split": None,  # optional split/fold filter applied while reading `train`
    "output": None,
    "device": "cpu",
    "dreams_checkpoint": None,  # DreaMS SSL Lightning checkpoint (Zenodo 10997887)
    "chemberta_model": CHEMBERTA,
    "chemberta_revision": None,
    "chemberta_from_config": False,
    "chemberta_checkpoint": None,
    "canonical_smiles": False,  # RDKit-canonicalise training SMILES (the `_ca` inputs)
    "seed": 1234,
    "epochs": 2,
    "max_steps": None,
    "batch_size": 32,
    "lr": 1e-4,
    "cond_dim": 2048,
    "mapper_hidden": 2048,
    "n_blocks": 8,
    "unfreeze_last": 2,
    "w_map": 5.0,
    "w_ortho": 1e-3,
    "w_con_mapped": 0.0,
    "log_every": 50,
}
PATHS = ("train", "output", "dreams_checkpoint", "chemberta_checkpoint")
# Upstream argparse values implied by this port; the mapped keys come from the config.
UPSTREAM_FIXED = {
    "mol_space": "chemberta",
    "no_gaussian": True,
    "fold": "train",
    "meta_json": None,
    "init_spec_from_scratch": False,
    "init_mol_from_scratch": False,
    "random_mapper_init": False,
    "unfreeze_after": 0,
    "unfreeze_mol_last": 0,
    "unfreeze_mol_after": 0,
    "amp": False,
    "early_stop": False,
    "num_workers": 0,
    "resume": None,
}
UPSTREAM_MAPPED = (
    "seed", "epochs", "max_steps", "batch_size", "lr", "cond_dim", "mapper_hidden", "n_blocks",
    "unfreeze_last", "w_map", "w_ortho", "w_con_mapped",
)
# Stored upstream arguments that cannot change `train_real` with the values above.
UPSTREAM_INERT = (
    "w_con", "w_supcon", "supcon_temp", "supcon_k", "allow_mol_update_in_con", "spec_bins",
    "fp_bits", "freeze_mol_adapter", "formula_vocab", "adduct_vocab", "charge_vocab",
    "chem_model_type", "steps", "demo", "cpu", "pin_memory", "persistent_workers", "log_every",
    "save_every", "val_every", "val_batches", "patience", "min_delta", "early_metric", "no_wandb",
    "outdir",
)


def load_config(path, **overrides):
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text()) or {}
    config = {**DEFAULTS, **raw, **{k: v for k, v in overrides.items() if v is not None}}
    for key in PATHS:
        if config[key] is not None and key not in overrides:
            value = Path(config[key]).expanduser()
            config[key] = str(value if value.is_absolute() else (path.parent / value).resolve())
    model = Path(config["chemberta_model"]).expanduser()
    if not model.is_absolute() and (path.parent / model).exists():
        config["chemberta_model"] = str((path.parent / model).resolve())
    return config


def upstream_args(config):
    """The upstream `train_real` arguments this configuration corresponds to."""
    return {
        **UPSTREAM_FIXED,
        **{key: config[key] for key in UPSTREAM_MAPPED},
        "chemberta_model": config["chemberta_model"],
        "chemberta_ckpt": config["chemberta_checkpoint"],
    }


def training_records(config, limit=None):
    records = load_queries(config["train"], split=config["split"])
    if limit is not None:
        records = records[:limit]
    if config["canonical_smiles"]:
        from rdkit import Chem

        for record in records:
            record["target_smiles"] = Chem.MolToSmiles(Chem.MolFromSmiles(record["target_smiles"]))
    return [(r["target_smiles"], r["spectrum"]) for r in records]


def train(config, *, limit=None):
    """Run upstream `train_real` for `config` and write `last.pt`; returns its path."""
    import torch

    from .model import SpecBridge, load_chemberta, load_dreams_checkpoint, pad_peaks

    device = torch.device(config["device"])
    seed = int(config["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    records = training_records(config, limit)

    def collate(batch):
        return [smiles for smiles, _ in batch], pad_peaks([spectrum for _, spectrum in batch])

    loader = torch.utils.data.DataLoader(
        records,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    )
    # Construction order matches upstream, so the global RNG initialises the
    # trainable modules identically: DreaMS, adapter, ChemBERTa, projections, mapper.
    dreams = load_dreams_checkpoint(config["dreams_checkpoint"], build_pretraining_head=True)
    model = SpecBridge(
        dreams,
        lambda: load_chemberta(
            config["chemberta_model"],
            revision=config["chemberta_revision"],
            from_config=bool(config["chemberta_from_config"]),
            checkpoint=config["chemberta_checkpoint"],
        ),
        cond_dim=int(config["cond_dim"]),
        hidden=int(config["mapper_hidden"]),
        n_blocks=int(config["n_blocks"]),
    ).to(device)
    # Upstream never calls model.train(): DreaMS keeps the eval mode it was loaded in
    # and ChemBERTa the mode its loader left it in; the new modules are in train mode.
    if int(config["unfreeze_last"]) > 0:
        model.spec.unfreeze_last(int(config["unfreeze_last"]))

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=float(config["lr"]), weight_decay=1e-4)
    total_steps = int(config["epochs"]) * len(loader)
    warmup = int(0.05 * total_steps)

    def lr_lambda(it):
        if it < warmup:
            return (it + 1) / max(1, warmup)
        t = (it - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    log = []
    step = 0
    max_steps = config["max_steps"]
    for _ in range(int(config["epochs"])):
        for smiles, peaks in loader:
            step += 1
            peaks = peaks.to(device)
            z_s = model.spec(peaks)
            z_m = model.embed_smiles(smiles, z_s.device)
            mu = model.mapB(z_s)
            loss, parts = model.align_losses(
                mu, z_m,
                w_map=float(config["w_map"]),
                w_ortho=float(config["w_ortho"]),
                w_con_mapped=float(config["w_con_mapped"]),
            )
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            if step % int(config["log_every"]) == 0 or step == 1:
                entry = {"step": step, "loss": loss.item(), **{k: v.item() for k, v in parts.items()}}
                log.append(entry)
                print(json.dumps(entry), flush=True)
            if max_steps and step >= int(max_steps):
                break
        if max_steps and step >= int(max_steps):
            break
    path = output / "last.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "opt": optimizer.state_dict(),
            "step": step,
            "args": upstream_args(config),
        },
        path,
    )
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Override the output directory")
    parser.add_argument("--device", help="Override the device, e.g. cuda:0")
    parser.add_argument("--max-steps", type=int, help="Stop after N optimizer steps")
    parser.add_argument("--limit", type=int, help="Use only the first N training spectra (smoke)")
    args = parser.parse_args(argv)
    config = load_config(
        args.config,
        output=str(args.output.resolve()) if args.output else None,
        device=args.device,
        max_steps=args.max_steps,
    )
    print(train(config, limit=args.limit))


if __name__ == "__main__":
    main()
