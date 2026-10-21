"""Explicit, offline SAFE tokenizer loading for FRIGID inference."""
import os
from pathlib import Path
from safe.tokenizer import SAFETokenizer


def get_tokenizer(hf_cache_dir=None):
    path = os.environ.get("FRIGID_SAFE_TOKENIZER_JSON")
    if not path or not Path(path).is_file():
        raise ValueError("FRIGID requires the safe_tokenizer asset")
    tokenizer = SAFETokenizer.from_pretrained(path, local_files_only=True).get_pretrained()
    tokenizer.add_tokens(["<", ">"])
    return tokenizer
