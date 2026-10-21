"""Rank candidates by ICEBERG-predicted spectra and native entropy distance."""

from ...simulators import iceberg as simulator
from ...simulators.retrieve import retrieve, validate_metadata


def validate(config, queries):
    for query in queries:
        validate_metadata(query)
    if config.parameters.get("query_intensity_space") not in {"native", "raw"}:
        raise ValueError("query_intensity_space must be specified explicitly")


def predict(config, queries):
    import torch

    torch.set_num_threads(config.workers)
    torch.manual_seed(config.seed)
    model = simulator.load_model(config)
    return retrieve(
        config, queries,
        lambda smiles, query: simulator.predict_spectrum(model, config, smiles, query),
        prediction_pooling="add",
    )
