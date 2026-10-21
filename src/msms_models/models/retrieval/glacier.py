"""Rank candidates by GLACIER-predicted spectra and native entropy distance."""

from ...simulators import glacier as simulator
from ...simulators.retrieve import retrieve


def predict(config, queries):
    import torch

    torch.set_num_threads(config.workers)
    torch.manual_seed(config.seed)
    model = simulator.load_model(config)
    return retrieve(
        config, queries, lambda smiles, query: simulator.predict_spectrum(model, config, smiles, query)
    )
