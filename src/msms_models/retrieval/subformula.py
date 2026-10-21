"""Compatibility import; nearest-neighbor fitting lives in methods/nearest_neighbor."""

import sys

from ..methods.nearest_neighbor import index as _implementation

sys.modules[__name__] = _implementation
