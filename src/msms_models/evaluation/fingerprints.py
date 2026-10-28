"""Spectrum-to-fingerprint quality: Tanimoto between a thresholded posterior and the target.

The encoder metric of Tables 1 and 4: each posterior is binarized at ``threshold`` (``>=``)
and compared with the target's Morgan radius-2/4096 fingerprint without chirality; the
mean is over spectra.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from ..encoders.mist import align_posterior


def fingerprint_tanimoto(queries, posterior, posterior_ids, threshold: float) -> dict:
    predicted = align_posterior(posterior, posterior_ids, queries) >= threshold
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096,
                                                          includeChirality=False)
    target = np.stack([generator.GetFingerprintAsNumPy(Chem.MolFromSmiles(q["target_smiles"]))
                       for q in queries]).astype(bool)
    intersection = (predicted & target).sum(1)
    union = (predicted | target).sum(1)
    values = intersection / np.maximum(union, 1)
    return {"n_spectra": len(queries), "threshold": threshold,
            "mean_tanimoto": float(values.mean())}
