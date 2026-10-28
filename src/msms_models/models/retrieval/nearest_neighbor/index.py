"""Sub-formula neighbour index built on training spectra, tuned on validation spectra."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from scipy import sparse

from ....data.io import write_json
from .vendor import features as native

GRID = {
    "w": (1.0, 2.0, 4.0, 8.0, 16.0),
    "alpha": (0.0, 0.25, 0.5, 1.0, 2.0, 4.0),
    "T": (0.01, 0.02, 0.05, 0.1, 0.2, 0.5),
    "k": (1, 3, 5, 10, 20, 50),
}


def _record(query):
    """A neighbour record: base-peak-normalized peaks and both formulas."""
    mz = query["spectrum"]["mz"]
    intensity = np.asarray(query["spectrum"]["intensity"], float)
    intensity = intensity * (100.0 / intensity.max())
    return {
        "formula": query["formula"],
        "formula_corrected": query["precursor_formula"],
        "peaks": [
            {"mz": float(m), "intensity_norm": float(i)} for m, i in zip(mz, intensity, strict=True)
        ],
    }


def tokenize(queries):
    tokens = []
    for query in queries:
        if len(native._HEAVY_CACHE) > 256:
            native._HEAVY_CACHE.clear()
        tokens.append(native.subformula_tokens(_record(query)))
    return tokens


def _fingerprints(queries):
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=4096, includeChirality=False
    )
    values, cache = [], {}
    for query in queries:
        smiles = query["target_smiles"]
        if smiles not in cache:
            cache[smiles] = generator.GetFingerprintAsNumPy(Chem.MolFromSmiles(smiles))
        values.append(cache[smiles])
    return np.stack(values).astype(np.uint8)


def _jaccard(prediction, target):
    return np.logical_and(prediction, target).sum(1) / np.maximum(
        np.logical_or(prediction, target).sum(1), 1
    )


class SubformulaIndex:
    def __init__(self, matrix, formulas, fingerprints, ids, vocabulary, parameters, metadata=None):
        self.matrix = matrix.tocsr()
        self.formulas = np.asarray(formulas, dtype=np.int32)
        self.fingerprints = np.asarray(fingerprints, dtype=np.uint8)
        self.ids = np.asarray(ids, dtype=str)
        self.vocabulary = vocabulary
        self.parameters = parameters
        self.metadata = metadata or {}
        self.unique_formulas, self.formula_inverse = np.unique(
            self.formulas, axis=0, return_inverse=True
        )

    @classmethod
    def build(cls, training):
        matrix, vocabulary = native.build_token_matrix(tokenize(training))
        return cls(
            matrix,
            native.formula_vectors([q["formula"] for q in training]),
            _fingerprints(training),
            [q["id"] for q in training],
            vocabulary,
            {},
        )

    def save(self, directory, *, metadata=None):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(directory / "train_tokens.npz", self.matrix)
        np.save(directory / "train_formulas.npy", self.formulas)
        np.save(directory / "fingerprints_packed.npy",
                np.packbits(self.fingerprints, axis=1, bitorder="little"))
        np.save(directory / "train_ids.npy", self.ids)
        write_json(directory / "vocabulary.json", self.vocabulary)
        write_json(directory / "index.json", {
            "parameters": self.parameters, "metadata": {**self.metadata, **(metadata or {})},
        })

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        index = json.loads((directory / "index.json").read_text())
        return cls(
            sparse.load_npz(directory / "train_tokens.npz"),
            np.load(directory / "train_formulas.npy"),
            np.unpackbits(np.load(directory / "fingerprints_packed.npy"), axis=1,
                          bitorder="little"),
            np.load(directory / "train_ids.npy"),
            json.loads((directory / "vocabulary.json").read_text()),
            index["parameters"],
            index["metadata"],
        )

    def _query_features(self, queries):
        matrix, _ = native.build_token_matrix(tokenize(queries), self.vocabulary)
        return matrix, native.formula_vectors([q["formula"] for q in queries])

    def _formula_score(self, query_formulas, weight):
        # Unique training formulas avoid repeatedly evaluating the same counts.
        values = native.weighted_formula_sim(query_formulas, self.unique_formulas, weight, batch=1)
        return values[:, self.formula_inverse]

    def predict(self, queries, *, batch_size=16):
        p = self.parameters
        matrix, formulas = self._query_features(queries)
        prediction = np.empty((len(queries), 4096), np.float32)
        for begin in range(0, len(queries), batch_size):
            end = min(begin + batch_size, len(queries))
            spectral = (matrix[begin:end] @ self.matrix.T).toarray().astype(np.float32)
            scores = spectral + np.float32(p["alpha"]) * self._formula_score(
                formulas[begin:end], p["w"]
            )
            index, score = native.topk_sorted(scores, int(p["k"]))
            weight = native.softmax_weights(score, p["T"], int(p["k"]))
            # Consensus fingerprints as float32 matrix-vector products.
            prediction[begin:end] = np.stack(
                [
                    w @ self.fingerprints[i].astype(np.float32)
                    for w, i in zip(weight, index, strict=True)
                ]
            )
        return prediction

    def select(self, validation, *, grid=None, batch_size=8):
        """Choose (w, alpha, T, k) by mean binary fingerprint Jaccard on validation spectra."""
        grid = grid or GRID
        ks = [int(k) for k in grid["k"] if k <= len(self.ids)]
        matrix, formulas = self._query_features(validation)
        truth = _fingerprints(validation)
        settings = [
            (float(w), float(a), float(t), k)
            for w in grid["w"]
            for a in grid["alpha"]
            for t in grid["T"]
            for k in ks
        ]
        sums = {setting: 0.0 for setting in settings}
        for begin in range(0, len(validation), batch_size):
            end = min(begin + batch_size, len(validation))
            spectral = (matrix[begin:end] @ self.matrix.T).toarray().astype(np.float32)
            for w in grid["w"]:
                formula = self._formula_score(formulas[begin:end], w)
                for alpha in grid["alpha"]:
                    if alpha == 0 and w != grid["w"][0]:
                        continue
                    index, score = native.topk_sorted(
                        spectral + np.float32(alpha) * formula, max(ks)
                    )
                    fp = self.fingerprints[index].astype(np.float32)
                    for temperature in grid["T"]:
                        for k in ks:
                            weight = native.softmax_weights(score, temperature, k)
                            binary = np.einsum("bk,bkc->bc", weight, fp[:, :k]) >= 0.5
                            value = float(_jaccard(binary, truth[begin:end]).sum())
                            sums[(float(w), float(alpha), float(temperature), k)] += value
        for setting in settings:
            w, alpha, temperature, k = setting
            if alpha == 0:
                sums[setting] = sums[(float(grid["w"][0]), alpha, temperature, k)]
        mean = {s: value / len(validation) for s, value in sums.items()}
        best = max(settings, key=lambda s: (round(mean[s], 6), -s[3], s[2], -s[1], -s[0]))
        self.parameters = dict(zip(("w", "alpha", "T", "k"), best, strict=True))
        return {
            "selected": self.parameters,
            "validation_rows": len(validation),
            "validation_jaccard": mean[best],
            "grid": {key: list(value) for key, value in grid.items()},
        }


def fit(training, validation, output, *, grid=None):
    """Build the index from training spectra and select its settings on validation spectra."""
    model = SubformulaIndex.build(training)
    selection = model.select(validation, grid=grid)
    model.save(output, metadata={"selection": selection})
    return selection
