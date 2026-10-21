"""Compatibility import for nearest-neighbor feature construction."""
import sys
from ..methods.nearest_neighbor import features as _implementation
sys.modules[__name__] = _implementation
