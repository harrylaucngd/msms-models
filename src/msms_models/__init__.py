"""Molecular retrieval and generation from MS/MS spectra; model runtimes load on demand."""

from importlib import import_module

__version__ = "0.2.0"

_EXPORTS = {
    "RunConfig": (".config", "RunConfig"),
    "predict": (".api", "predict"),
    "run": (".api", "run"),
    "evaluate": (".evaluation.metrics", "evaluate"),
    "MetricProtocol": (".evaluation.metrics", "MetricProtocol"),
    "get_model": (".models", "get_model"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module, __name__), attribute)
    globals()[name] = value
    return value
