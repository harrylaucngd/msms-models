"""Spectrum and molecule featurization of the MassSpecGym baselines (``data/transforms.py``)."""

from __future__ import annotations

import re

import numpy as np

# The 118 elements in MassSpecGym's order (``definitions.CHEM_ELEMS``).
ELEMENTS = (
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl",
    "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As",
    "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In",
    "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb",
    "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl",
    "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk",
    "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh",
    "Fl", "Mc", "Lv", "Ts", "Og",
)


def preprocess_peaks(mz, intensity, *, mz_from=10.0, mz_to=1000.0, n_max_peaks=60):
    """matchms ``select_by_mz``, ``reduce_to_number_of_peaks`` and ``normalize_intensities``.

    matchms keeps the ``n_max_peaks`` most intense peaks with NumPy's default argsort; the
    NumPy 1.24 order that MassSpecGym ran with is reproduced for ties.
    """
    from ..models.retrieval.fingerprint import numpy124_argsort

    mz, intensity = np.asarray(mz, dtype=float), np.asarray(intensity, dtype=float)
    keep = (mz_from <= mz) & (mz <= mz_to)
    mz, intensity = mz[keep], intensity[keep]
    if n_max_peaks is not None and intensity.size >= n_max_peaks:
        top = np.array(numpy124_argsort(intensity.tolist())[-n_max_peaks:], dtype=int)
        top = top[np.argsort(mz[top])]
        mz, intensity = mz[top], intensity[top]
    if intensity.size:
        intensity = intensity / intensity.max()
    return mz, intensity


def tokenize_spectrum(query, *, n_peaks=60, max_mz=1000.0, precursor_intensity=1.1):
    """``SpecTokenizer``: the precursor as a first peak, then (m/z, intensity) rows, zero-padded.

    Returns an ``(n_peaks + 1, 2)`` array.
    """
    mz, intensity = preprocess_peaks(
        query["spectrum"]["mz"], query["spectrum"]["intensity"], mz_to=max_mz, n_max_peaks=n_peaks
    )
    rows = np.zeros((n_peaks + 1, 2))
    rows[0] = query["precursor_mz"], precursor_intensity
    rows[1:len(mz) + 1, 0], rows[1:len(mz) + 1, 1] = mz, intensity
    return rows


def bin_spectrum(query, *, max_mz=1005.0, bin_width=1.0):
    """``SpecBinner``: relative intensities summed in ``max_mz / bin_width`` bins."""
    mz, intensity = preprocess_peaks(
        query["spectrum"]["mz"], query["spectrum"]["intensity"], mz_to=max_mz, n_max_peaks=None
    )
    num_bins = int(np.ceil(max_mz / bin_width))
    binned = np.zeros(num_bins)
    np.add.at(binned, np.clip(np.floor(mz / bin_width).astype(int), 0, num_bins - 1), intensity)
    return binned / binned.max()


def morgan_fingerprint(smiles, size=4096):
    """``MolFingerprinter``: radius-2 Morgan bits, as ``GetMorganFingerprintAsBitVect``."""
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=size)
    return generator.GetFingerprintAsNumPy(Chem.MolFromSmiles(smiles)).astype(np.int32)


def formula_vector(formula):
    """``MolToFormulaVector``: element counts, hydrogens included, in MassSpecGym's order."""
    vector = np.zeros(len(ELEMENTS), dtype=np.int32)
    for element, count in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        vector[ELEMENTS.index(element)] += int(count or 1)
    return vector
