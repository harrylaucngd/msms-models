"""Compatibility entry points; implementations are owned by their methods."""

from importlib import import_module

from ..simulation import canonical_candidate, score_candidates, validate_metadata

__all__ = [
    "canonical_candidate", "score_candidates", "validate_metadata",
    "load_forward", "forward_spectrum", "predict",
]


def _implementation(config):
    method = "glacier" if config.model == "glacier" else "iceberg"
    return import_module(f"msms_models.methods.{method}.inference")


def load_forward(config):
    return _implementation(config).load_model(config)


def forward_spectrum(model, config, smiles, query):
    return _implementation(config).predict_spectrum(model, config, smiles, query)


def predict(config, queries):
    return _implementation(config).predict(config, queries)
