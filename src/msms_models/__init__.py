"""Molecular inference and evaluation; optional backends are imported on demand."""

from importlib import import_module

__version__ = "0.2.0"

_EXPORTS = {
    "RunConfig": (".config", "RunConfig"),
    "infer": (".api", "infer"),
    "predict": (".api", "predict"),
    "evaluate": (".metrics", "evaluate"),
    "MetricProtocol": (".metrics", "MetricProtocol"),
    "get_model": (".registry", "get_model"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module, __name__), attribute)
    globals()[name] = value
    return value
