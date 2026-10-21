"""chem_utils.py

msms-models: the subset of DiffMS ``src/mist/utils/chem_utils.py`` (coleygroup/DiffMS
@9d9f4fd) used by the MIST peak-formula featurizer and encoder. Definitions are
verbatim; unused helpers and the RDKit mass tables (needed only for the unused
fragment m/z vector) are omitted.
"""

import re
from collections import defaultdict

import numpy as np

CHEM_FORMULA_SIZE = "([A-Z][a-z]*)([0-9]*)"

VALID_ELEMENTS = [
    "C",
    "H",
    "As",
    "B",
    "Br",
    "Cl",
    "Co",
    "F",
    "Fe",
    "I",
    "K",
    "N",
    "Na",
    "O",
    "P",
    "S",
    "Se",
    "Si",
]

ELEMENT_VECTORS = np.eye(len(VALID_ELEMENTS))

# Reasonable normalization vector for elements
# Estimated by max counts (+ 1 when zero)
NORM_VEC = np.array([81, 158, 2, 1, 3, 10, 1, 17, 1, 6, 1, 19, 2, 34, 6, 6, 2, 6])

element_to_position = dict(zip(VALID_ELEMENTS, ELEMENT_VECTORS))

ION_LST = [
    "[M+H]+",
    "[M+Na]+",
    "[M+K]+",
    "[M-H2O+H]+",
    "[M+H3N+H]+",
    "[M]+",
    "[M-H4O2+H]+",
]

ion_to_idx = dict(zip(ION_LST, np.arange(len(ION_LST))))

instrument_to_type = defaultdict(lambda : "unknown")
instrument_to_type.update({
    "Thermo Finnigan Velos Orbitrap": "orbitrap",
    "Thermo Finnigan Elite Orbitrap": "orbitrap",
    "Orbitrap Fusion Lumos": "orbitrap",
    "Q-ToF (LCMS)": "qtof",
    "Unknown (LCMS)": "unknown",
    "ion trap": "iontrap",
    "FTICR (LCMS)": "fticr",
    "Bruker Q-ToF (LCMS)": "qtof",
    "Orbitrap (LCMS)": "orbitrap",
})

instruments = sorted(list(set(instrument_to_type.values())))
max_instr_idx = len(instruments) + 1
instrument_to_idx = dict(zip(instruments, np.arange(len(instruments))))


def get_ion_idx(ionization: str) -> int:
    """map ionization to its index in one hot encoding"""
    return ion_to_idx[ionization]


def get_instr_idx(instrument: str) -> int:
    """map instrument to its index in one hot encoding"""
    inst = instrument_to_type.get(instrument, "unknown")
    return instrument_to_idx[inst]


def formula_to_dense(chem_formula: str) -> np.ndarray:
    """formula_to_dense.

    Args:
        chem_formula (str): Input chemical formal
    Return:
        np.ndarray of vector

    """
    total_onehot = []
    for (chem_symbol, num) in re.findall(CHEM_FORMULA_SIZE, chem_formula):
        # Convert num to int
        num = 1 if num == "" else int(num)
        one_hot = element_to_position[chem_symbol].reshape(1, -1)
        one_hot_repeats = np.repeat(one_hot, repeats=num, axis=0)
        total_onehot.append(one_hot_repeats)

    # Check if null
    if len(total_onehot) == 0:
        dense_vec = np.zeros(len(element_to_position))
    else:
        dense_vec = np.vstack(total_onehot).sum(0)

    return dense_vec
