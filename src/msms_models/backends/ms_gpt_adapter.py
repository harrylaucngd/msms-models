"""Compatibility import for the MS-GPT adapter."""

import sys

from ..methods.ms_gpt import adapter as _implementation

sys.modules[__name__] = _implementation
