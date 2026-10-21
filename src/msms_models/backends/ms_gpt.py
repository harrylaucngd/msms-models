"""Compatibility import; implementation lives in methods/ms_gpt/inference.py."""

import sys

from ..methods.ms_gpt import inference as _implementation

sys.modules[__name__] = _implementation
