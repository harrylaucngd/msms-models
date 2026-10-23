"""Train the Morgan-4096 decoder from a prepared SQLite database."""

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np

from .data import FoldDataset
from .inference import MODEL_CONFIG


def run_epoch(model, loader, device, optimizer=None):
    import torch
    from torch.nn import functional as F

    model.train(optimizer is not None)
    losses = []
    with torch.set_grad_enabled(optimizer is not None):
        for fp, formula, tokens, labels in loader:
            fp, formula, tokens, labels = [v.to(device) for v in (fp, formula, tokens, labels)]
            if optimizer is not None:
                optimizer.zero_grad()
            logits, hydrogen = model({'FP': fp, 'MF': formula, 'tokens_X': tokens})
            loss = F.cross_entropy(logits.transpose(1, 2).float(), labels)
            if optimizer is not None:
                loss = loss + 0.03 * F.mse_loss(hydrogen.float(), formula[:, -1:])
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training/validation loss')
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            losses.append(loss.detach().item())
    if not losses:
        raise ValueError('Empty training/validation loader')
    return float(np.mean(losses))


def train(args):
    import torch
    from torch.utils.data import DataLoader

    from .vendor.model_msnovelist import TranscoderModel

    if args.epochs < 1 or args.batch_size < 2 or args.workers < 0:
        raise ValueError('epochs >= 1, batch-size >= 2 and workers >= 0 are required')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    training = FoldDataset(args.database, args.fold)
    validation = FoldDataset(args.database, args.fold, validation=True)
    if len(training) % args.batch_size == 1:
        raise ValueError('Choose a batch-size that does not leave a singleton BatchNorm batch')
    device = torch.device(args.device)
    model = TranscoderModel(config=MODEL_CONFIG).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    train_loader = DataLoader(training, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers)
    val_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers)
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    resolved = {**vars(args), 'database': str(Path(args.database).expanduser().resolve()),
                'output': str(output), 'model_config': MODEL_CONFIG,
                'loss_weight_sm': 1.0, 'loss_weight_H': 0.03, 'fingerprint_simulation': False}
    (output / 'config.json').write_text(json.dumps(resolved, indent=2) + '\n')
    for epoch in range(args.epochs):
        loss = run_epoch(model, train_loader, device, optimizer)
        val_loss = run_epoch(model, val_loader, device)
        torch.save(model.state_dict(), output / f'w-{epoch + 1}-{loss:.3f}-{val_loss:.3f}.pth')
        torch.save(optimizer.state_dict(), output / f'optimizer-{epoch + 1}.pth')
        logging.info('epoch=%d train=%.6f validation_smiles=%.6f', epoch + 1, loss, val_loss)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', required=True)
    parser.add_argument('--output', required=True, help='New output directory (must not exist)')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    train(args)


if __name__ == '__main__':
    main()
