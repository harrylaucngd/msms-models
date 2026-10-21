"""Compatibility import; implementation lives in methods/nearest_neighbor/inference.py."""

import sys

from ..methods.nearest_neighbor import inference as _implementation

sys.modules[__name__] = _implementation
