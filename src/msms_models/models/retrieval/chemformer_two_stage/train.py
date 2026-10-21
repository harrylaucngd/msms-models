"""Train GLMR: contrastive pre-retrieval, top-K precomputation, generative stage.

Ported from GLMR ``src/train_pre_retrieval.py``, ``src/precompute_candidates.py`` and
``src/train_generative.py``. Configurations keep GLMR's ``data/model/training/eval``
layout, so the ``config`` stored in each released checkpoint is a valid input. The
seed, module construction order, DataLoader settings and validation schedule all draw
from the same global RNG streams as the original and are kept unchanged.

Two behaviours that differ between the released checkpoints are explicit options:

* ``training.accumulation_steps`` (generative stage, default 1).
* ``training.legacy_scaled_clip`` (generative stage, default false). GLMR at HEAD
  unscales gradients before clipping their norm to 1. The corrected checkpoint was
  trained with the earlier loop, which clipped the loss-scaled gradients
  (no ``scaler.unscale_``); ``true`` selects that loop. On CPU, where the scaler is
  disabled, the two are identical.

::

    python -m msms_models.models.retrieval.chemformer_two_stage.train config \\
        --stage pre-retrieval --variant corrected --output pre.yaml --mgf ... --candidates ...
    python -m msms_models.models.retrieval.chemformer_two_stage.train pre-retrieval --config pre.yaml
    python -m msms_models.models.retrieval.chemformer_two_stage.train precompute \\
        --config pre.yaml --pre-retrieval-checkpoint .../best_model.pt --output-dir precomputed
    python -m msms_models.models.retrieval.chemformer_two_stage.train generative --config gen.yaml
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import time
from pathlib import Path

import numpy as np

VARIANTS = ("inflated", "corrected")
STAGES = ("pre-retrieval", "generative")

# Chemformer molecular encoder and spectral encoder sizes (identical in every run).
_MODEL = {
    "d_model": 512,
    "vocab_size": 523,
    "enc_num_layers": 6,
    "num_heads": 8,
    "d_feedforward": 2048,
    "dropout": 0.1,
    "activation": "gelu",
    "max_seq_len": 512,
    "pad_token_idx": 0,
    "spec_d_model": 512,
    "spec_num_layers": 6,
    "spec_num_heads": 8,
    "spec_d_feedforward": 2048,
    "spec_dropout": 0.1,
}
_EVAL_PRE = {"top_k": [1, 5, 20], "mces_threshold": 15}
_EVAL_GEN = {"beam_size": 5, "max_gen_len": 512, "top_k": [1, 5, 20], "pre_retrieval_k": 40,
             "mces_threshold": 15}
# Keys that name files; everything else is compared with the released checkpoints.
PATH_KEYS = {
    "data": ("mgf_path", "candidates_path", "train_candidates_path", "val_candidates_path"),
    "model": ("chemformer_ckpt", "pre_retrieval_ckpt"),
    "training": ("save_dir",),
}
PORT_KEYS = {"training": ("legacy_scaled_clip",)}


def default_config(stage: str, variant: str, *, mgf=None, candidates=None, chemformer=None,
                   pre_retrieval=None, precomputed_dir=None, save_dir=None) -> dict:
    """The configuration of a released checkpoint, with the file paths supplied here.

    ``inflated`` is the run on the original MassSpecGym SMILES strings (``outputs``);
    ``corrected`` the run on RDKit-canonical SMILES (``outputs_new``).
    """
    if stage not in STAGES or variant not in VARIANTS:
        raise ValueError(f"stage must be one of {STAGES} and variant one of {VARIANTS}")
    model = {"chemformer_ckpt": chemformer, **_MODEL}
    if stage == "pre-retrieval":
        return {
            "data": {"mgf_path": mgf, "candidates_path": candidates, "max_peaks": 61,
                     "max_smiles_len": 512, "noise_scale": 0.3},
            "model": {**model, "temperature": 0.1},
            "training": {"epochs": 300, "batch_size": 256, "lr": 1.0e-4, "weight_decay": 0.1,
                         "num_workers": 4, "val_every_n_epochs": 5, "save_dir": save_dir,
                         "seed": 42},
            "eval": copy.deepcopy(_EVAL_PRE),
        }
    precomputed = Path(precomputed_dir) if precomputed_dir else None
    training = {"epochs": 30, "batch_size": 128}
    if variant == "inflated":
        training["accumulation_steps"] = 4
    training |= {"lr": 1.0e-4, "weight_decay": 0.1, "num_workers": 4,
                 "val_every_n_epochs": 1 if variant == "inflated" else 5,
                 "save_dir": save_dir, "seed": 42}
    if variant == "corrected":
        training["legacy_scaled_clip"] = True
    return {
        "data": {
            "mgf_path": mgf,
            "candidates_path": candidates,
            "train_candidates_path": str(precomputed / "train_top40.pkl") if precomputed else None,
            "val_candidates_path": str(precomputed / "val_top40.pkl") if precomputed else None,
            "max_peaks": 61, "max_smiles_len": 512, "top_k": 40,
        },
        "model": {"chemformer_ckpt": chemformer, "pre_retrieval_ckpt": pre_retrieval,
                  **_MODEL},
        "training": training,
        "eval": copy.deepcopy(_EVAL_GEN),
    }


def logic(cfg: dict) -> dict:
    """``cfg`` without file paths and port-only options (for comparing with checkpoints)."""
    out = copy.deepcopy(cfg)
    for groups in (PATH_KEYS, PORT_KEYS):
        for section, keys in groups.items():
            for key in keys:
                out.get(section, {}).pop(key, None)
    return out


def set_seed(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _device(device=None):
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def _autocast(torch, device):
    return torch.autocast(device_type="cuda", enabled=device.type == "cuda")


def _to(batch, device):
    import torch

    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def _loader(dataset, cfg, collate, shuffle):
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=shuffle,
        num_workers=cfg["training"]["num_workers"],
        collate_fn=collate,
        pin_memory=True,
        drop_last=shuffle,
    )


def _validate_every(cfg, epoch):
    epochs = cfg["training"]["epochs"]
    return (epoch + 1) % cfg["training"]["val_every_n_epochs"] == 0 or epoch == epochs - 1


def train_pre_retrieval(cfg: dict, *, resume=None, device=None, log=print) -> float:
    """Stage 1: train the spectral encoder against frozen Chemformer CLS embeddings."""
    import torch

    from .data import ContrastiveDataset, collate_contrastive, parse_mgf, split_records
    from .model import (
        ChemformerEncoder,
        PreRetrievalModel,
        SpectralEncoder,
        load_chemformer_checkpoint,
    )
    from .tokenizer import SmilesTokenizer

    set_seed(cfg["training"]["seed"])
    device = _device(device)
    save_dir = Path(cfg["training"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = SmilesTokenizer()
    train_recs, val_recs, _ = split_records(parse_mgf(cfg["data"]["mgf_path"]))
    log(f"Train: {len(train_recs)}, Val: {len(val_recs)}")
    datasets = [
        ContrastiveDataset(records, tokenizer, max_peaks=cfg["data"]["max_peaks"],
                           max_smiles_len=cfg["data"]["max_smiles_len"], augment_intensity=True,
                           noise_scale=cfg["data"]["noise_scale"])
        for records in (train_recs, val_recs)
    ]
    train_loader = _loader(datasets[0], cfg, collate_contrastive, shuffle=True)
    val_loader = _loader(datasets[1], cfg, collate_contrastive, shuffle=False)

    hp, state = load_chemformer_checkpoint(cfg["model"]["chemformer_ckpt"])
    mol_encoder = ChemformerEncoder.from_chemformer(hp, state)
    mol_encoder.eval()
    spec_encoder = SpectralEncoder.from_config(cfg["model"])
    model = PreRetrievalModel(mol_encoder, spec_encoder, cfg["model"]["temperature"]).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg["training"]["lr"],
                                  weight_decay=cfg["training"]["weight_decay"])

    start_epoch, best = 0, float("inf")
    if resume and os.path.exists(resume):
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.spec_encoder.load_state_dict(ckpt["spec_encoder_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch, best = ckpt["epoch"] + 1, ckpt.get("best_val_loss", float("inf"))

    def save(name, epoch):
        torch.save({
            "epoch": epoch,
            "spec_encoder_state_dict": model.spec_encoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best,
            "config": cfg,
        }, save_dir / name)

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    epochs = cfg["training"]["epochs"]
    for epoch in range(start_epoch, epochs):
        model.spec_encoder.train()
        model.mol_encoder.eval()
        total, batches, started = 0.0, 0, time.time()
        for batch in train_loader:
            batch = _to(batch, device)
            optimizer.zero_grad()
            with _autocast(torch, device):
                loss = model(batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()
            batches += 1
        message = (f"Epoch {epoch + 1:3d}/{epochs} | Loss: {total / max(batches, 1):.4f} | "
                   f"{time.time() - started:.0f}s")
        if _validate_every(cfg, epoch):
            model.spec_encoder.eval()
            val_total, val_batches = 0.0, 0
            with torch.no_grad():
                for batch in val_loader:
                    with _autocast(torch, device):
                        val_total += model(_to(batch, device)).item()
                    val_batches += 1
            val_loss = val_total / max(val_batches, 1)
            if val_loss < best:
                best = val_loss
                save("best_model.pt", epoch)
            message += f" | ValLoss: {val_loss:.4f} | Best: {best:.4f}"
        log(message)
        if (epoch + 1) % 50 == 0:
            save(f"ckpt_ep{epoch + 1}.pt", epoch)
    save("final_model.pt", epochs - 1)
    return best


def precompute_candidates(cfg: dict, pre_retrieval_checkpoint, output_dir, *, top_k=40,
                          device=None, log=print) -> dict:
    """Top-K stage-1 candidates for every training and validation spectrum.

    Writes ``{split}_top{top_k}.pkl`` mapping the record's index within its split to
    its candidate SMILES. As in GLMR, every unique SMILES of the candidate file (all
    splits) is embedded in sorted order, 512 per batch.
    """
    import pickle

    import torch

    from .data import load_pickle, parse_mgf, split_records
    from .model import (
        ChemformerEncoder,
        PreRetrievalModel,
        SpectralEncoder,
        load_chemformer_checkpoint,
    )
    from .tokenizer import SmilesTokenizer

    device = _device(device)
    tokenizer = SmilesTokenizer()
    train_recs, val_recs, _ = split_records(parse_mgf(cfg["data"]["mgf_path"]))
    candidates = load_pickle(cfg["data"]["candidates_path"])
    hp, state = load_chemformer_checkpoint(cfg["model"]["chemformer_ckpt"])
    mol_encoder = ChemformerEncoder.from_chemformer(hp, state)
    spec_encoder = SpectralEncoder.from_config(cfg["model"])
    model = PreRetrievalModel(mol_encoder, spec_encoder, cfg["model"]["temperature"])
    ckpt = torch.load(pre_retrieval_checkpoint, map_location=device, weights_only=False)
    model.spec_encoder.load_state_dict(ckpt["spec_encoder_state_dict"])
    model.to(device)
    model.eval()

    from .data import batch_tokens

    unique = sorted({s for pool in candidates.values() for s in pool})
    log(f"Encoding {len(unique)} unique candidate molecules")
    embeddings = {}
    for start in range(0, len(unique), 512):
        chunk = unique[start : start + 512]
        ids, mask = batch_tokens([tokenizer.encode(s) for s in chunk])
        ids, mask = ids.to(device), mask.to(device)
        with torch.no_grad(), _autocast(torch, device):
            values = model.encode_molecules(ids, mask)
        for j, s in enumerate(chunk):
            embeddings[s] = values[j].cpu().numpy()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    max_peaks = cfg["data"]["max_peaks"]
    for split, records in (("train", train_recs), ("val", val_recs)):
        result, skipped = {}, 0
        for index, record in enumerate(records):
            pool = candidates.get(record["SMILES"])
            valid = [s for s in pool if s in embeddings] if pool is not None else []
            if not valid:
                skipped += 1
                continue
            mzs, intensities = record["mzs"], record["intensities"]
            if len(mzs) > max_peaks:
                keep = np.argsort(intensities)[-max_peaks:]
                mzs, intensities = mzs[keep], intensities[keep]
            order = np.argsort(mzs)
            spectrum = torch.tensor(
                np.stack([mzs[order], intensities[order]], axis=-1), dtype=torch.float32
            ).unsqueeze(0).to(device)
            mask = torch.ones(1, len(order), dtype=torch.bool, device=device)
            with torch.no_grad(), _autocast(torch, device):
                query = model.encode_spectra(spectrum, mask).cpu().numpy()[0]
            similarity = np.stack([embeddings[s] for s in valid]) @ query
            ranking = np.argsort(-similarity)
            result[index] = [valid[ranking[i]] for i in range(min(top_k, len(ranking)))]
        path = output_dir / f"{split}_top{top_k}.pkl"
        with open(path, "wb") as handle:
            pickle.dump(result, handle)
        log(f"{split}: {len(result)} records, skipped {skipped} -> {path}")
        written[split] = path
    return written


def train_generative(cfg: dict, *, resume=None, device=None, log=print) -> float:
    """Stage 2: train cross-fusion and the Chemformer decoder with frozen encoders."""
    import torch

    from .data import (
        GenerativeDataset,
        collate_generative,
        load_pickle,
        parse_mgf,
        split_records,
    )
    from .model import (
        ChemformerDecoder,
        ChemformerEncoder,
        CrossFusion,
        GenerativeRetrievalModel,
        SpectralEncoder,
        load_chemformer_checkpoint,
    )
    from .tokenizer import SmilesTokenizer

    set_seed(cfg["training"]["seed"])
    device = _device(device)
    save_dir = Path(cfg["training"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = SmilesTokenizer()
    train_recs, val_recs, _ = split_records(parse_mgf(cfg["data"]["mgf_path"]))
    precomputed = [load_pickle(cfg["data"][k]) for k in
                   ("train_candidates_path", "val_candidates_path")]
    datasets = [
        GenerativeDataset(records, cands, tokenizer, max_peaks=cfg["data"]["max_peaks"],
                          max_smiles_len=cfg["data"]["max_smiles_len"], top_k=cfg["data"]["top_k"])
        for records, cands in zip((train_recs, val_recs), precomputed)
    ]
    train_loader = _loader(datasets[0], cfg, collate_generative, shuffle=True)
    val_loader = _loader(datasets[1], cfg, collate_generative, shuffle=False)
    log(f"Train: {len(datasets[0])}, Val: {len(datasets[1])}")

    # Construction order as in GLMR: encoder, spectral encoder, cross-fusion, decoder.
    hp, state = load_chemformer_checkpoint(cfg["model"]["chemformer_ckpt"])
    mol_encoder = ChemformerEncoder.from_chemformer(hp, state)
    mol_encoder.eval()
    spec_encoder = SpectralEncoder.from_config(cfg["model"])
    pre = torch.load(cfg["model"]["pre_retrieval_ckpt"], map_location="cpu", weights_only=False)
    spec_encoder.load_state_dict(pre["spec_encoder_state_dict"])
    spec_encoder.eval()
    cross_fusion = CrossFusion.from_config(cfg["model"])
    decoder = ChemformerDecoder.from_chemformer(hp, state, encoder=mol_encoder)
    model = GenerativeRetrievalModel(mol_encoder, spec_encoder, cross_fusion, decoder,
                                     pad_token_idx=cfg["model"]["pad_token_idx"]).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    accumulation = cfg["training"].get("accumulation_steps", 1)
    legacy_clip = bool(cfg["training"].get("legacy_scaled_clip", False))
    log(f"Gradient accumulation: {accumulation}; clip scaled gradients: {legacy_clip}")
    optimizer = torch.optim.AdamW(trainable, lr=cfg["training"]["lr"],
                                  weight_decay=cfg["training"]["weight_decay"])

    start_epoch, best = 0, float("inf")
    if resume and os.path.exists(resume):
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.cross_fusion.load_state_dict(ckpt["cross_fusion_state_dict"])
        model.decoder.load_state_dict(ckpt["decoder_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch, best = ckpt["epoch"] + 1, ckpt.get("best_val_loss", float("inf"))

    def save(name, epoch):
        torch.save({
            "epoch": epoch,
            "cross_fusion_state_dict": model.cross_fusion.state_dict(),
            "decoder_state_dict": model.decoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best,
            "config": cfg,
        }, save_dir / name)

    epochs = cfg["training"]["epochs"]
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    for epoch in range(start_epoch, epochs):
        model.cross_fusion.train()
        model.decoder.train()
        model.mol_encoder.eval()
        model.spec_encoder.eval()
        total, batches, started = 0.0, 0, time.time()
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            batch = _to(batch, device)
            with _autocast(torch, device):
                loss = model.train_step(batch)
                scaled = loss / accumulation
            scaler.scale(scaled).backward()
            if (step + 1) % accumulation == 0 or (step + 1) == len(train_loader):
                if not legacy_clip:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            total += loss.item()
            batches += 1
        message = (f"Epoch {epoch + 1:3d}/{epochs} | Loss: {total / max(batches, 1):.4f} | "
                   f"{time.time() - started:.0f}s")
        if _validate_every(cfg, epoch):
            model.eval()
            val_total, val_batches = 0.0, 0
            with torch.no_grad():
                for batch in val_loader:
                    with _autocast(torch, device):
                        val_total += model.train_step(_to(batch, device)).item()
                    val_batches += 1
            val_loss = val_total / max(val_batches, 1)
            if val_loss < best:
                best = val_loss
                save("best_gen_model.pt", epoch)
            message += f" | ValLoss: {val_loss:.4f} | Best: {best:.4f}"
        log(message)
        if (epoch + 1) % 10 == 0:
            save(f"gen_ckpt_ep{epoch + 1}.pt", epoch)
    save("final_gen_model.pt", epochs - 1)
    return best


def _read(path):
    import yaml

    return yaml.safe_load(Path(path).read_text())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    write = sub.add_parser("config", help="Write the configuration of a released checkpoint")
    write.add_argument("--stage", choices=STAGES, required=True)
    write.add_argument("--variant", choices=VARIANTS, required=True)
    write.add_argument("--output", type=Path, required=True)
    for name in ("mgf", "candidates", "chemformer", "pre-retrieval", "precomputed-dir",
                 "save-dir"):
        write.add_argument(f"--{name}")
    for name in ("pre-retrieval", "generative"):
        stage = sub.add_parser(name, help=f"Train the {name} stage")
        stage.add_argument("--config", type=Path, required=True)
        stage.add_argument("--resume")
        stage.add_argument("--device")
    pre = sub.add_parser("precompute", help="Top-K stage-1 candidates for train and val")
    pre.add_argument("--config", type=Path, required=True)
    pre.add_argument("--pre-retrieval-checkpoint", required=True)
    pre.add_argument("--output-dir", type=Path, required=True)
    pre.add_argument("--top-k", type=int, default=40)
    pre.add_argument("--device")
    args = parser.parse_args(argv)

    if args.command == "config":
        import yaml

        cfg = default_config(
            args.stage, args.variant, mgf=args.mgf, candidates=args.candidates,
            chemformer=args.chemformer, pre_retrieval=args.pre_retrieval,
            precomputed_dir=args.precomputed_dir, save_dir=args.save_dir,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(yaml.safe_dump(cfg, sort_keys=False))
    elif args.command == "pre-retrieval":
        train_pre_retrieval(_read(args.config), resume=args.resume, device=args.device)
    elif args.command == "generative":
        train_generative(_read(args.config), resume=args.resume, device=args.device)
    else:
        precompute_candidates(_read(args.config), args.pre_retrieval_checkpoint, args.output_dir,
                              top_k=args.top_k, device=args.device)


if __name__ == "__main__":
    main()
