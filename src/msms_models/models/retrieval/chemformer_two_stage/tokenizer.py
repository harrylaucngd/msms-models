"""Regex SMILES tokenizer with Chemformer's 523-token BART vocabulary.

The vocabulary reproduces the layout of Chemformer's ``bart_vocab.json`` that the
pretrained ``step=1000000.ckpt`` embedding matrix was trained with: six special
tokens (pad=0, unk=1, start/CLS=2, end=3, mask=4, sep=5), 60 LogD and 6 property
tokens of the molecular-optimisation vocabulary, blank placeholders up to index 271,
and 251 chemical tokens from index 272. The regular expression is Chemformer's
SMILES pattern. Ported unchanged from GLMR ``src/data/tokenizer.py``.
"""

from __future__ import annotations

import json
import re

SMILES_REGEX = (
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p"
    r"|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
)

PAD_IDX = 0
UNK_IDX = 1
START_IDX = 2  # also the CLS position of the encoder
END_IDX = 3
MASK_IDX = 4
SEP_IDX = 5
VOCAB_SIZE = 523

_SPECIAL = [" ", "?", "^", "&", " ", " "]
_LOGD_POSITIVE = [
    "(-0.1, 0.1]", "(0.1, 0.3]", "(0.3, 0.5]", "(0.5, 0.7]",
    "(0.7, 0.9]", "(0.9, 1.1]", "(1.1, 1.3]", "(1.3, 1.5]",
    "(1.5, 1.7]", "(1.7, 1.9]", "(1.9, 2.1]", "(2.1, 2.3]",
    "(2.3, 2.5]", "(2.5, 2.7]", "(2.7, 2.9]", "(2.9, 3.1]",
    "(3.1, 3.3]", "(3.3, 3.5]", "(3.5, 3.7]", "(3.7, 3.9]",
    "(3.9, 4.1]", "(4.1, 4.3]", "(4.3, 4.5]", "(4.5, 4.7]",
    "(4.7, 4.9]", "(4.9, 5.1]", "(5.1, 5.3]", "(5.3, 5.5]",
    "(5.5, 5.7]", "(5.7, 5.9]", "(5.9, inf]",
]
_LOGD_NEGATIVE = [
    "(-0.3, -0.1]", "(-0.5, -0.3]", "(-0.7, -0.5]", "(-0.9, -0.7]",
    "(-1.1, -0.9]", "(-1.3, -1.1]", "(-1.5, -1.3]", "(-1.7, -1.5]",
    "(-1.9, -1.7]", "(-2.1, -1.9]", "(-2.3, -2.1]", "(-2.5, -2.3]",
    "(-2.7, -2.5]", "(-2.9, -2.7]", "(-3.1, -2.9]", "(-3.3, -3.1]",
    "(-3.5, -3.3]", "(-3.7, -3.5]", "(-3.9, -3.7]", "(-4.1, -3.9]",
    "(-4.3, -4.1]", "(-4.5, -4.3]", "(-4.7, -4.5]", "(-4.9, -4.7]",
    "(-5.1, -4.9]", "(-5.3, -5.1]", "(-5.5, -5.3]", "(-5.7, -5.5]",
    "(-inf, -5.7]",
]
_PROPERTY = [
    "Solubility_low->high", "Solubility_high->low", "Solubility_no_change",
    "Clint_low->high", "Clint_high->low", "Clint_no_change",
]
_CHEMICAL = [
    "C", "c", "1", "(", "-", "2", "s", "N", "=", ")", "n", "Br", "3",
    "O", "[nH]", "[C@H]", "/", "\\", "[C@@H]", "[C@]", "o", "4",
    "[C@@]", "5", "6", "7", "[n+]", ".", "[Br-]", "S", "I", "F", "Cl",
    "[O-]", "#", "P", "[Na+]", "[N+]", "[Cl-]", "8", "[Si]", "[N-]",
    "[18F]", "9", "[I-]", "B", "[S+]", "[2H]", "[P+]", "[125I]",
    "%10", "%11", "[n-]", "[B-]", "[O]", "[o+]", "[N@@+]", "[N@+]",
    "[PH]", "[se]", "[Se]", "[s+]", "[Li+]", "[P@]", "[P@@]", "[3H]",
    "[K+]", "[OH-]", "[te+]", "[se+]", "[Te]", "[S@+]", "[BH2-]",
    "[11CH3]", "[11C]", "[Cl+3]", "[N@@]", "[S-]", "[C+]", "[P@+]",
    "[C-]", "[Zn+2]", "[Ca+2]", "[Mg+2]", "[SeH]", "[BH-]", "[SH2]",
    "[TeH2]", "[SiH4]", "[N@]", "[14CH2]", "[Ag]", "[S@@+]", "[I+]",
    "[MgH2]", "[125IH]", "[Se+]", "[As]", "[SiH2]", "[Ra]", "[IH2]",
    "[P-]", "[Na]", "[NH-]", "[Cs]", "[Zn]", "[Li]", "[As+]", "[te]",
    "[131I]", "[Ag+]", "p", "[Al-3]", "[Rb+]", "[13CH]", "[11CH]",
    "[11CH2]", "[14c]", "[14C@@]", "[127I]", "[Mg]", "[14C]", "[123I]",
    "[124I]", "[F-]", "[14CH3]", "[135I]", "%12", "[NH+]", "[76Br]",
    "[32PH]", "[35S]", "b", "[73Se]", "[11C@@H]", "[Se-]", "[c+]",
    "[14C@H]", "[18OH]", "[SH]", "[S@@]", "[Cs+]", "[He]", "[O+]",
    "[nH+]", "[NH2+]", "[32P]", "[Zn+]", "[BH3-]", "[I+3]", "[Si-]",
    "[SH+]", "[19F]", "[Br+2]", "[I+2]", "[Al+3]", "[123I-]",
    "[131I-]", "[127Xe]", "[133Xe]", "[89Sr+2]", "[N]", "[82Rb+]",
    "[75Se]", "[Rb]", "[81Kr]", "[18F-]", "[13NH3]", "[K]", "[Cl+2]",
    "[Zn-2]", "[SeH2]", "[AsH3]", "[Kr]", "[Xe]", "[S@]", "[cH-]",
    "[NH4+]", "[Al]", "[Si@]", "[15n]", "[SH-]", "[SiH]", "[11c]",
    "[OH]", "[c-]", "[18FH]", "[123IH]", "[13c]", "[13cH]", "[14cH]",
    "[Cl+]", "[Sr+2]", "[CH-]", "[Bi]", "[B]", "[Ba+2]", "[Bi+3]",
    "[SiH-]", "[b-]", "[H+]", "[13C]", "[OH+]", "[14CH]", "[15nH]",
    "[CaH2]", "[LiH]", "[C]", "[Ca]", "[42K+]", "[123Te]", "[S-2]",
    "[223Ra+2]", "[S]", "[Ra+2]", "[22Na+]", "[125I-]", "[85Sr+2]",
    "[PH2]", "[SrH2]", "[15OH2]", "[47Ca+2]", "[85SrH2]", "[45Ca+2]",
    "[B@@-]", "[B@-]", "[17F]", "[PH2+]", "[11C-]", "[Mg+]", "[NaH]",
    "[P@@+]", "[SiH3-]", "[P-3]", "[KH]", "[Be+2]", "[NH3+]",
    "[Ag-4]", "[18O]", "[14C@@H]", "[CH]", "[CH2]", "[O-2]",
    "[124I-]", "[As-]", "[Ba]", "[223Ra]", "[82Rb]", "[76BrH]",
    "[AsH]", "[131Cs]",
]


def default_vocabulary() -> list[str]:
    """The 523-token vocabulary of the pretrained Chemformer checkpoint."""
    logd = [f"LogD_change_{r}" for r in _LOGD_POSITIVE + _LOGD_NEGATIVE]
    placeholders = [" "] * (272 - len(_SPECIAL) - len(logd) - len(_PROPERTY))
    vocab = _SPECIAL + logd + _PROPERTY + placeholders + _CHEMICAL
    return vocab


class SmilesTokenizer:
    """Regex-based SMILES tokenizer matching Chemformer's vocabulary."""

    def __init__(self, vocab_path: str | None = None):
        self.regex = re.compile(SMILES_REGEX)
        self.pad_idx, self.unk_idx = PAD_IDX, UNK_IDX
        self.start_idx, self.end_idx = START_IDX, END_IDX
        if vocab_path is not None:
            with open(vocab_path) as handle:
                vocab = json.load(handle)["vocabulary"]
        else:
            vocab = default_vocabulary()
        # Blank placeholders repeat; as in the original, the last index wins.
        self.token2idx = {token: i for i, token in enumerate(vocab)}
        self.idx2token = dict(enumerate(vocab))
        self.vocab_size = len(vocab)

    def tokenize(self, smiles: str) -> list[str]:
        return self.regex.findall(smiles)

    def encode(self, smiles: str, max_len: int = 512, add_special: bool = True) -> list[int]:
        ids = [self.token2idx.get(t, self.unk_idx) for t in self.tokenize(smiles)]
        if add_special:
            ids = [self.start_idx, *ids, self.end_idx]
        return ids[:max_len]

    def decode(self, ids) -> str:
        # GLMR skips pad, start and end only; the mask token (4) decodes to a blank.
        skip = {self.pad_idx, self.start_idx, self.end_idx}
        return "".join(self.idx2token.get(i, "?") for i in ids if i not in skip)
