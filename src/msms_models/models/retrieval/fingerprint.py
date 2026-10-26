"""Rank candidates by Tanimoto similarity to a predicted Morgan4096 fingerprint."""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


def rank_fingerprints(queries, probabilities, candidates, *, threshold=None, ties="stable"):
    """Continuous Tanimoto against each candidate's Morgan r2/4096 fingerprint.

    ``ties="legacy-quicksort"`` reproduces the tie order of the original MIST evaluation,
    a reversed NumPy 1.24 argsort; ``"stable"`` keeps the supplied candidate order on ties.
    """
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if threshold is not None:
        probabilities = (probabilities >= threshold).astype(np.float32)
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=4096, includeChirality=False
    )
    cache, rows = {}, []
    for query, vector in zip(queries, probabilities, strict=True):
        strings = candidates[query["id"]]
        for smiles in strings:
            if smiles not in cache:
                fingerprint = generator.GetFingerprintAsNumPy(Chem.MolFromSmiles(smiles))
                cache[smiles] = np.packbits(fingerprint, bitorder="little")
        matrix = np.unpackbits(
            np.stack([cache[s] for s in strings]), axis=1, bitorder="little"
        ).astype(np.float32)
        intersection = matrix @ vector
        scores = intersection / np.maximum(vector.sum() + matrix.sum(1) - intersection, 1e-8)
        order = (np.argsort(-scores, kind="stable") if ties == "stable"
                 else np.array(numpy124_argsort(scores.tolist())[::-1]))
        rows.append({"id": query["id"], "candidates": [strings[i] for i in order],
                     "scores": scores[order].astype(float).tolist(), "status": "ok"})
    return rows


def numpy124_argsort(values):
    """``np.argsort(values)`` as NumPy 1.24 computes it, on any NumPy version and CPU.

    Later releases dispatch argsort to SIMD sorts that order ties differently, so the
    introsort of NumPy 1.24 (npysort/quicksort.cpp) is reproduced here: median-of-three
    partitioning, insertion sort below 16 elements, heapsort past the depth limit. NaN
    sorts last.
    """
    v = values

    def lt(a, b):
        return a < b or (b != b and a == a)

    def heapsort(order, start, n):
        a = [None, *order[start:start + n]]  # 1-based, as in NumPy
        for low in range(n >> 1, 0, -1):
            tmp, i, j = a[low], low, low << 1
            while j <= n:
                if j < n and lt(v[a[j]], v[a[j + 1]]):
                    j += 1
                if not lt(v[tmp], v[a[j]]):
                    break
                a[i], i, j = a[j], j, j + j
            a[i] = tmp
        while n > 1:
            tmp, a[n] = a[n], a[1]
            n -= 1
            i, j = 1, 2
            while j <= n:
                if j < n and lt(v[a[j]], v[a[j + 1]]):
                    j += 1
                if not lt(v[tmp], v[a[j]]):
                    break
                a[i], i, j = a[j], j, j + j
            a[i] = tmp
        order[start:start + len(a) - 1] = a[1:]

    order = list(range(len(v)))
    if len(order) < 2:
        return order
    left, right, stack = 0, len(order) - 1, []
    depth = 2 * (len(order).bit_length() - 1)
    while True:
        if depth < 0:
            heapsort(order, left, right - left + 1)
        else:
            while right - left > 15:
                mid = left + ((right - left) >> 1)
                if lt(v[order[mid]], v[order[left]]):
                    order[mid], order[left] = order[left], order[mid]
                if lt(v[order[right]], v[order[mid]]):
                    order[right], order[mid] = order[mid], order[right]
                if lt(v[order[mid]], v[order[left]]):
                    order[mid], order[left] = order[left], order[mid]
                pivot, i, j = v[order[mid]], left, right - 1
                order[mid], order[j] = order[j], order[mid]
                while True:
                    i += 1
                    while lt(v[order[i]], pivot):
                        i += 1
                    j -= 1
                    while lt(pivot, v[order[j]]):
                        j -= 1
                    if i >= j:
                        break
                    order[i], order[j] = order[j], order[i]
                order[i], order[right - 1] = order[right - 1], order[i]
                depth -= 1
                if i - left < right - i:  # continue with the smaller side
                    stack.append((i + 1, right, depth))
                    right = i - 1
                else:
                    stack.append((left, i - 1, depth))
                    left = i + 1
            for i in range(left + 1, right + 1):
                item, j = order[i], i
                while j > left and lt(v[item], v[order[j - 1]]):
                    order[j] = order[j - 1]
                    j -= 1
                order[j] = item
        if not stack:
            return order
        left, right, depth = stack.pop()
