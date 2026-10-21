# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Serena Khoo
# Adapted from serenaklm/ML_MS_analysis, revision 6817e9bd994f.
# Only formula annotation, sparse features and consensus functions are retained.

import itertools
import math
import re

import numpy as np
from scipy import sparse

MONO_MASS = {
    "H": 1.00782503207, "D": 2.01410177812, "He": 4.00260325415, "Li": 7.01600455, "Be": 9.0121822,
    "B": 11.0093054, "C": 12.0, "N": 14.0030740048, "O": 15.99491461956, "F": 18.99840322,
    "Ne": 19.9924401754, "Na": 22.9897692809, "Mg": 23.985041700, "Al": 26.98153863,
    "Si": 27.9769265325, "P": 30.97376163, "S": 31.97207100, "Cl": 34.96885268, "Ar": 39.9623831225,
    "K": 38.96370668, "Ca": 39.96259098, "Fe": 55.9349375, "Zn": 63.9291422, "As": 74.9215965,
    "Se": 79.9165213, "Br": 78.9183371, "I": 126.904473,
}
ELECTRON_MASS = 0.00054857990946
ELEMENTS = sorted(MONO_MASS)
ELEMENT_IDX = {e: i for i, e in enumerate(ELEMENTS)}
HETEROATOM = np.array([e not in ("C", "H") for e in ELEMENTS])
HALOGENS = ("F", "Cl", "Br", "I")
FORMULA_RE = re.compile(r"([A-Z][a-z]?)(\d*)")

def parse_formula(formula):
    """'C6H12O6' -> {'C': 6, 'H': 12, 'O': 6}; charge signs are ignored."""
    counts = {}
    for element, count in FORMULA_RE.findall(formula):
        if element:
            counts[element] = counts.get(element, 0) + (int(count) if count else 1)
    return counts

def check_known_elements(counts, formula=None):
    """Raise a readable error for elements this module has no monoisotopic mass for. `ELEMENTS` (and with it the
    element-count vectors and the sub-formula enumeration) is derived from MONO_MASS, so an unlisted element --
    a metal, say -- would otherwise surface as a bare KeyError deep inside the annotation."""
    unknown = sorted(e for e in counts if e not in MONO_MASS)
    if unknown:
        where = f" in formula {formula!r}" if formula is not None else ""
        raise ValueError(f"nn_lib has no monoisotopic mass for {', '.join(unknown)}{where}. Add the element(s) to "
                         f"nn_lib.MONO_MASS (ELEMENTS and ELEMENT_IDX are derived from it) to use this dataset.")

def counts_to_string(counts):
    """Hill-order string of an element-count dict."""
    parts = []
    for e in ["C", "H"] + [x for x in sorted(counts) if x not in ("C", "H")]:
        c = counts.get(e, 0)
        if c > 0:
            parts.append(e if c == 1 else f"{e}{c}")
    return "".join(parts)

def formula_vectors(formulas):
    """(n x len(ELEMENTS)) int32 element-count matrix."""
    E = np.zeros((len(formulas), len(ELEMENTS)), dtype=np.int32)
    for i, f in enumerate(formulas):
        counts = parse_formula(f)
        check_known_elements(counts, f)
        for e, c in counts.items():
            E[i, ELEMENT_IDX[e]] = c
    return E

def weighted_formula_sim(test_E, train_E, hetero_weight, batch=64):
    """Heteroatom-weighted Bray-Curtis similarity of element-count vectors, in [0, 1]:
        s_F(q, t) = 1 - sum_e w_e |q_e - t_e| / sum_e w_e (q_e + t_e),   w_e = hetero_weight for e not in {C, H}, else 1.
    hetero_weight = 1 is the plain Bray-Curtis similarity; 1 iff the formulas are identical."""
    w = np.ones(len(ELEMENTS), dtype=np.float32)
    w[HETEROATOM] = hetero_weight
    out = np.empty((test_E.shape[0], train_E.shape[0]), dtype=np.float32)
    test_wsum = (test_E * w).sum(axis=1).astype(np.float32)
    train_wsum = (train_E * w).sum(axis=1).astype(np.float32)
    for s in range(0, test_E.shape[0], batch):
        e = min(s + batch, test_E.shape[0])
        d = (np.abs(test_E[s:e, None, :] - train_E[None, :, :]) * w[None, None, :]).sum(axis=2)
        out[s:e] = 1.0 - d / (test_wsum[s:e, None] + train_wsum[None, :])
    return out

_HEAVY_CACHE = {}

def heavy_subformulas(parent_counts):
    """All sub-formulas of the parent's heavy (non-H) atoms: (elements, count matrix, sorted monoisotopic masses)."""
    key = counts_to_string(parent_counts)
    if key in _HEAVY_CACHE:
        return _HEAVY_CACHE[key]
    check_known_elements(parent_counts, key)
    elements = [e for e in parent_counts if e != "H" and parent_counts[e] > 0]
    if elements:
        combos = np.array(list(itertools.product(*[range(parent_counts[e] + 1) for e in elements])), dtype=np.int16)
        masses = combos.astype(np.float64) @ np.array([MONO_MASS[e] for e in elements], dtype=np.float64)
    else:
        combos, masses = np.zeros((1, 0), dtype=np.int16), np.zeros(1)
    order = np.argsort(masses)
    out = (elements, combos[order], masses[order])
    _HEAVY_CACHE[key] = out
    return out

def annotate_peaks(mz_array, parent_counts, ppm=20.0, min_da=0.01):
    """For every peak m/z, the sub-formula of the precursor ion formula whose (singly charged, positive) ion mass is
    closest to it within max(min_da, ppm): heavy atoms are enumerated, the hydrogen count is solved from the mass
    residual, and a loose valence check (2 * RDBE >= -1) is applied. None when no sub-formula explains the peak."""
    elements, combos, heavy_masses = heavy_subformulas(parent_counts)
    H_parent = parent_counts.get("H", 0)
    mH = MONO_MASS["H"]
    c_idx = elements.index("C") if "C" in elements else None
    n_idx = elements.index("N") if "N" in elements else None
    p_idx = elements.index("P") if "P" in elements else None
    x_idx = [elements.index(x) for x in HALOGENS if x in elements]
    out = []
    for mz in mz_array:
        target = mz + ELECTRON_MASS
        tol = max(min_da, ppm * 1e-6 * target)
        lo = np.searchsorted(heavy_masses, target - H_parent * mH - tol)
        hi = np.searchsorted(heavy_masses, target + tol)
        if hi <= lo:
            out.append(None); continue
        hm = heavy_masses[lo:hi]
        H = np.rint((target - hm) / mH).astype(np.int32)
        err = np.abs(target - hm - H * mH)
        ok = (H >= 0) & (H <= H_parent) & (err <= tol)
        if ok.any():
            cand = combos[lo:hi]
            C = cand[:, c_idx].astype(np.int32) if c_idx is not None else 0
            N = cand[:, n_idx].astype(np.int32) if n_idx is not None else 0
            P = cand[:, p_idx].astype(np.int32) if p_idx is not None else 0
            X = sum(cand[:, i].astype(np.int32) for i in x_idx) if x_idx else 0
            ok &= (2 * C + N + P + 2 - (H + X)) >= -1
        if not ok.any():
            out.append(None); continue
        best = np.where(ok)[0][np.argmin(err[ok])]
        counts = {e: int(c) for e, c in zip(elements, combos[lo + best]) if c > 0}
        if H[best] > 0:
            counts["H"] = int(H[best])
        out.append(counts)
    return out

def subformula_tokens(rec, top_k=100, ppm=20.0, min_da=0.01):
    """{token: weight}: a fragment token 'F:<sub-formula>' and a neutral-loss token 'L:<ion formula - sub-formula>'
    per annotated peak (the top_k most intense), weighted by sqrt(intensity_norm) (max over repeats)."""
    parent = parse_formula(rec.get("formula_corrected") or rec["formula"])
    peaks = sorted(rec["peaks"], key=lambda p: -p["intensity_norm"])[:top_k]
    annotations = annotate_peaks(np.array([p["mz"] for p in peaks], dtype=np.float64), parent, ppm=ppm, min_da=min_da)
    tokens = {}
    for p, ann in zip(peaks, annotations):
        if ann is None:
            continue
        w = math.sqrt(max(p["intensity_norm"], 0.0))
        frag = "F:" + counts_to_string(ann)
        loss_counts = {e: parent.get(e, 0) - ann.get(e, 0) for e in parent}
        loss = "L:" + (counts_to_string(loss_counts) if any(v > 0 for v in loss_counts.values()) else "none")
        tokens[frag] = max(tokens.get(frag, 0.0), w)
        tokens[loss] = max(tokens.get(loss, 0.0), w)
    return tokens


def build_token_matrix(token_dicts, vocab=None):
    """Sparse CSR matrix of L2-normalised token rows. The vocabulary is built from `token_dicts` when not given
    (training set); tokens outside the vocabulary are dropped."""
    if vocab is None:
        vocab = {}
        for d in token_dicts:
            for t in d:
                if t not in vocab:
                    vocab[t] = len(vocab)
    rows, cols, vals = [], [], []
    for i, d in enumerate(token_dicts):
        for t, w in d.items():
            j = vocab.get(t)
            if j is not None:
                rows.append(i); cols.append(j); vals.append(w)
    M = sparse.csr_matrix((vals, (rows, cols)), shape=(len(token_dicts), len(vocab)), dtype=np.float32)
    norms = np.sqrt(np.asarray(M.multiply(M).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    return (sparse.diags(1.0 / norms) @ M).tocsr(), vocab

# --------------------------------------------------------------------------------------------------
# Top-k consensus
# --------------------------------------------------------------------------------------------------

def topk_sorted(score, K, batch=512):
    """Per row of `score`: indices and values of the K largest entries, best first."""
    n = score.shape[0]
    idx = np.empty((n, K), dtype=np.int64)
    sc = np.empty((n, K), dtype=np.float32)
    for s in range(0, n, batch):
        Sb = score[s:s + batch]
        part = np.argpartition(-Sb, K - 1, axis=1)[:, :K]
        vals = np.take_along_axis(Sb, part, axis=1)
        order = np.argsort(-vals, axis=1)
        idx[s:s + batch] = np.take_along_axis(part, order, axis=1)
        sc[s:s + batch] = np.take_along_axis(vals, order, axis=1)
    return idx, sc

def softmax_weights(sc, T, k):
    """w_i proportional to exp((s_i - s_1) / T) over the k best scores (rows of `sc` are sorted best first)."""
    w = np.exp((sc[:, :k] - sc[:, :1]) / np.float32(T))
    return w / w.sum(axis=1, keepdims=True)

def softmax_consensus(idx, sc, train_FP, T, k, batch=256):
    """Prediction = elementwise threshold at 0.5 of the softmax(T)-weighted average of the top-k training fingerprints.
    k = 1 copies the fingerprint of the best match."""
    n = idx.shape[0]
    pred = np.empty((n, train_FP.shape[1]), dtype=np.uint8)
    for s in range(0, n, batch):
        w = softmax_weights(sc[s:s + batch], T, k)
        fp = train_FP[idx[s:s + batch, :k]].astype(np.float32)
        pred[s:s + batch] = (np.einsum("bk,bkc->bc", w, fp) >= 0.5)
    return pred
