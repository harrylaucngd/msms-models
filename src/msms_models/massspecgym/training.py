"""The training loop of the MassSpecGym baselines, as MassSpecGym's ``scripts/run.py`` set it up.

Lightning's behaviour is kept: Adam without weight decay, validation ``1 / val_check_interval``
times per epoch, early stopping when the validation loss has not improved for ``patience``
validations, and the checkpoint with the lowest validation loss. MassSpecGym trained on 8
GPUs with DDP; ``batch_size`` here is the global batch (8 × its per-GPU batch), which averages
gradients the same way on one device.
"""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path

import numpy as np


def seed_everything(seed):
    """``pytorch_lightning.seed_everything``."""
    import torch

    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fit(model, train, validation, loss_fn, *, settings, output, collate=None, device="cpu",
        log=print):
    """Train ``model`` and save its best state to ``output/model.pt`` with ``settings``.

    ``loss_fn(model, batch, device)`` returns the mean loss of a batch. ``settings`` must
    hold ``batch_size``, ``lr``, ``max_epochs``, ``val_check_interval``, ``patience`` and
    ``seed``; it is saved with the weights so that prediction rebuilds the same model.
    """
    import torch
    from torch.utils.data import DataLoader

    seed_everything(settings["seed"])
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["lr"], weight_decay=0.0)
    train_loader = DataLoader(train, batch_size=settings["batch_size"], shuffle=True,
                              collate_fn=collate)
    val_loader = DataLoader(validation, batch_size=settings["batch_size"], shuffle=False,
                            collate_fn=collate)
    # Lightning validates every int(interval × batches) training batches.
    every = max(1, int(settings["val_check_interval"] * len(train_loader)))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    best, waited, step = math.inf, 0, 0
    for epoch in range(settings["max_epochs"]):
        for index, batch in enumerate(train_loader, 1):
            model.train()
            optimizer.zero_grad()
            loss = loss_fn(model, batch, device)
            loss.backward()
            optimizer.step()
            step += 1
            if index % every:
                continue
            model.eval()
            with torch.no_grad():
                # The logged validation loss is the mean of the batch losses.
                val_loss = float(np.mean([loss_fn(model, b, device).item() for b in val_loader]))
            log(json.dumps({"epoch": epoch, "step": step, "train_loss": loss.item(),
                            "val_loss": val_loss}))
            if val_loss < best:
                best, waited = val_loss, 0
                torch.save({"state_dict": model.state_dict(), "settings": settings,
                            "step": step, "val_loss": val_loss}, output / "model.pt")
            else:
                waited += 1
                if waited >= settings["patience"]:
                    return best
    return best


def load(path, build, device="cpu"):
    """The model and settings saved by ``fit``; ``build(settings)`` makes the network."""
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = build(checkpoint["settings"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint["settings"]
