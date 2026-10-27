"""Layers of the MassSpecGym baselines."""

from __future__ import annotations

from .vendor.fourier_features import FourierFeatures  # noqa: F401


def mlp(in_channels, hidden_channels, out_channels, num_layers, dropout=0.0):
    """PyTorch Geometric's ``MLP`` as MassSpecGym builds it (version 2.5.3, ``norm=None``).

    Each hidden layer is Linear → ReLU → Dropout; the last layer is a plain Linear.
    """
    from torch import nn

    sizes = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
    layers = []
    for index, (size_in, size_out) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(nn.Linear(size_in, size_out))
        if index < num_layers - 1:
            layers += [nn.ReLU(), nn.Dropout(dropout)]
    return nn.Sequential(*layers)
