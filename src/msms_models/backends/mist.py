"""Compatibility import; implementation lives in methods/mist/inference.py."""

import sys

from ..methods.mist import inference as _implementation

sys.modules[__name__] = _implementation
