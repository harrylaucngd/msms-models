"""Compatibility import; implementation lives in methods/frigid/inference.py."""

import sys

from ..methods.frigid import inference as _implementation

sys.modules[__name__] = _implementation
