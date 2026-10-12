"""Lossless, fixed-vocabulary SMILES tokenization.

The decoder is randomly initialized, so it does not need compatibility with a
pretrained molecular vocabulary.  A printable-ASCII character vocabulary is
small, deterministic across every cascade corpus, and represents rare legal
atoms/charges without the irreversible ``<unk>`` failures present in the old
MolForge SentencePiece model.
"""

from __future__ import annotations

import re
from typing import Iterable


SMILES_TOKEN_PATTERN = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|"
    r"\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
)


def tokenize_smiles(smiles: str) -> list[str]:
    """Split a SMILES string with the canonical MolForge lexical regex."""
    tokens = SMILES_TOKEN_PATTERN.findall(smiles)
    if "".join(tokens) != smiles:
        raise ValueError(f"SMILES tokenizer did not cover the full string: {smiles!r}")
    return tokens


def remove_stereo(smiles: str) -> str:
    """Remove atom and bond stereo syntax while preserving connectivity."""
    tokens = tokenize_smiles(smiles)
    normalized = []
    for token in tokens:
        if token in {"/", "\\"}:
            continue
        normalized.append(token.replace("@", ""))
    return "".join(normalized)


class SmilesTokenizer:
    """Printable-ASCII character tokenizer with fixed reserved ids."""

    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    unk_token_id = 3
    _alphabet = tuple(chr(value) for value in range(33, 127))

    def __init__(self):
        self._token_to_id = {
            token: index + 4 for index, token in enumerate(self._alphabet)
        }
        self._id_to_token = {
            index + 4: token for index, token in enumerate(self._alphabet)
        }

    def __len__(self) -> int:
        return len(self._alphabet) + 4

    def encode_pretokenized(self, text: str) -> list[int]:
        compact = "".join(text.strip().split())
        return self.encode(compact)

    def encode(self, smiles: str) -> list[int]:
        normalized = remove_stereo(smiles)
        try:
            return [self._token_to_id[character] for character in normalized]
        except KeyError as error:
            raise ValueError(f"non-ASCII SMILES character {error.args[0]!r}") from error

    def decode(self, token_ids: Iterable[int], **_: object) -> str:
        output = []
        for value in token_ids:
            value = int(value)
            if value in {self.pad_token_id, self.bos_token_id, self.eos_token_id}:
                continue
            if value == self.unk_token_id or value not in self._id_to_token:
                raise ValueError(f"cannot decode unknown SMILES token id {value}")
            output.append(self._id_to_token[value])
        return "".join(output)
