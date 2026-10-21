"""Discover method specifications without importing optional model runtimes."""

from importlib import import_module

from .methods._types import Model

METHODS = ("nearest_neighbor", "mist", "molforge", "ms_gpt", "frigid", "foam", "iceberg", "glacier")
MODELS: dict[str, Model] = {}
for method in METHODS:
    for model in import_module(f".methods.{method}.spec", __package__).MODELS:
        if model.name in MODELS:
            raise ValueError(f"Duplicate model registration: {model.name}")
        MODELS[model.name] = model

DATASET_ALIASES = {"msg": "msg", "massspecgym": "msg", "canopus": "canopus", "nplib1": "canopus"}


def dataset_name(value: str) -> str:
    try:
        return DATASET_ALIASES[value.lower()]
    except KeyError:
        raise ValueError(f"Unknown dataset {value!r}; choose msg or canopus") from None


def get_model(name: str) -> Model:
    try:
        return MODELS[name]
    except KeyError:
        raise ValueError(f"Unknown model {name!r}; choose from {', '.join(MODELS)}") from None
