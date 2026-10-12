"""Chemistry utilities with graph/spectrum helpers imported only when needed."""
from importlib import import_module
from .chem_utils import *


def __getattr__(name):
    # Explicit routing avoids importing DGL for formula/fingerprint-only DLM use.
    if name.startswith('__'):
        raise AttributeError(name)
    for module_name in ('fingerprint', 'parallel_utils', 'misc_utils', 'denoising_utils'):
        module = import_module(f'{__name__}.{module_name}')
        if hasattr(module, name):
            value = getattr(module, name)
            globals()[name] = value
            return value
    raise AttributeError(name)
