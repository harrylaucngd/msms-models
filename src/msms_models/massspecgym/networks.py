"""The retrieval networks of the MassSpecGym baselines (``models/retrieval/*.py``)."""

from __future__ import annotations

import torch
from torch import nn

from .layers import FourierFeatures, mlp


class DeepSets(nn.Module):
    """``DeepSetsRetrieval``: per-peak φ, a sum over peaks, then ρ to a fingerprint.

    As upstream, the zero-padded peak rows pass through φ and are summed with the others.
    With ``fourier_features``, m/z is expanded by DreaMS Fourier features and projected to
    80% of the hidden width, and intensity is projected to the rest.
    """

    def __init__(self, hidden_channels=128, out_channels=4096, num_layers_per_mlp=4,
                 dropout=0.1, fourier_features=True):
        super().__init__()
        in_channels = 2
        self.fourier_features = fourier_features
        if fourier_features:
            self.ff = FourierFeatures()
            mz_channels = int(0.8 * hidden_channels)
            self.ff_proj_mz = nn.Linear(self.ff.num_features, mz_channels)
            self.ff_proj_i = nn.Linear(1, hidden_channels - mz_channels)
            in_channels = hidden_channels
        self.phi = mlp(in_channels, hidden_channels, hidden_channels, num_layers_per_mlp, dropout)
        self.rho = mlp(hidden_channels, hidden_channels, out_channels, num_layers_per_mlp, dropout)

    def forward(self, x):
        if self.fourier_features:
            x_mz = self.ff_proj_mz(self.ff(x[:, :, 0].unsqueeze(-1)))
            x_i = self.ff_proj_i(x[:, :, 1].unsqueeze(-1))
            x = torch.cat((x_mz, x_i), dim=-1)
        return torch.sigmoid(self.rho(self.phi(x).sum(dim=-2)))


class FingerprintFFN(nn.Module):
    """``FingerprintFFNRetrieval``: a binned spectrum through an MLP, then a sigmoid."""

    def __init__(self, in_channels=1005, hidden_channels=128, out_channels=4096, num_layers=7,
                 dropout=0.1):
        super().__init__()
        self.ffn = mlp(in_channels, hidden_channels, out_channels, num_layers, dropout)

    def forward(self, x):
        return torch.sigmoid(self.ffn(x))
