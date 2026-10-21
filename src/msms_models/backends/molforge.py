"""Compatibility import; implementation lives in methods/molforge/inference.py."""

import sys

from ..methods.molforge import inference as _implementation

sys.modules[__name__] = _implementation
