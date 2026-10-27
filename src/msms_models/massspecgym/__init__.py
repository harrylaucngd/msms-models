"""Featurization, layers and training shared by the MassSpecGym baselines.

The baselines of MassSpecGym (Bushuiev et al., NeurIPS 2024) are reimplemented without the
``massspecgym`` package, following ``pluskal-lab/MassSpecGym`` at the commits that produced
the published numbers: ``444395af2c`` (DeepSets), ``9e6bb0f52b`` (DeepSets with Fourier
features), ``e5080d6126`` (Fingerprint FFN) and ``485deb22df`` (SMILES/SELFIES Transformer,
transforms). Lightning, PyTorch Geometric and matchms are replaced by the few operations
the baselines use, with the same results.
"""
