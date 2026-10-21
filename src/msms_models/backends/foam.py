"""Compatibility import; implementation lives in methods/foam/inference.py."""

import sys

from ..methods.foam import inference as _implementation

sys.modules[__name__] = _implementation
