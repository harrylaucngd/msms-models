"""Portable, train-only sub-formula neighbour indices with validation selection."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from scipy import sparse

from .._vendor import subformula as native
from ..chemistry import inchikey14
from ..io import atomic_json

GRID = {
    "w": (1.0, 2.0, 4.0, 8.0, 16.0),
    "alpha": (0.0, 0.25, 0.5, 1.0, 2.0, 4.0),
    "T": (0.01, 0.02, 0.05, 0.1, 0.2, 0.5),
    "k": (1, 3, 5, 10, 20, 50),
}


def _record(query):
    if not query.get("formula") or not query.get("precursor_formula"):
        raise ValueError(f"{query['id']}: formula and precursor_formula are required")
    spectrum = query.get("spectrum", {})
    mz, intensity = spectrum.get("mz", []), np.asarray(spectrum.get("intensity", []), float)
    if not len(mz) or len(mz) != len(intensity) or intensity.max() <= 0:
        raise ValueError(f"{query['id']}: a nonempty measured spectrum is required")
    intensity = intensity * (100.0 / intensity.max())
    for formula in [query["formula"], query["precursor_formula"]]:
        counts = native.parse_formula(formula)
        if not counts:
            raise ValueError(f"Invalid formula {formula!r}")
        native.check_known_elements(counts, formula)
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
        smiles = query.get("target_smiles")
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None:
            raise ValueError(f"{query['id']}: valid training/validation target_smiles is required")
        if smiles not in cache:
            cache[smiles] = generator.GetFingerprintAsNumPy(mol)
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
        n = len(self.ids)
        if len(set(self.ids)) != n or self.matrix.shape[0] != n:
            raise ValueError("Invalid training index row identities")
        if self.fingerprints.shape != (n, 4096):
            raise ValueError("The training index must contain Morgan4096 fingerprints")
        if self.formulas.shape != (n, len(native.ELEMENTS)):
            raise ValueError("The training index has incompatible element vectors")
        if self.matrix.shape[1] != len(vocabulary) or set(vocabulary.values()) != set(
            range(len(vocabulary))
        ):
            raise ValueError("The token vocabulary is not a complete column mapping")
        if not np.isfinite(self.matrix.data).all() or (self.matrix.data < 0).any():
            raise ValueError("Invalid training token matrix")
        self.unique_formulas, self.formula_inverse = np.unique(
            self.formulas, axis=0, return_inverse=True
        )

    @classmethod
    def build(cls, training):
        if not training or any(q.get("split") not in (None, "train") for q in training):
            raise ValueError("The neighbour library must contain training queries only")
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
        destination = Path(directory)
        if destination.exists():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
        try:
            sparse.save_npz(directory / "train_tokens.npz", self.matrix)
            np.save(directory / "train_formulas.npy", self.formulas)
            np.save(
                directory / "fingerprints_packed.npy",
                np.packbits(self.fingerprints, axis=1, bitorder="little"),
            )
            np.save(directory / "train_ids.npy", self.ids)
            atomic_json(directory / "vocabulary.json", self.vocabulary)
            atomic_json(
                directory / "manifest.json",
                {
                    "schema_version": 1,
                    "method": "subformula-only",
                    "training_rows": len(self.ids),
                    "fingerprint": "morgan-r2-4096-achiral",
                    "parameters": self.parameters,
                    "elements": native.ELEMENTS,
                    "metadata": {**self.metadata, **(metadata or {})},
                    "annotation": {
                        "max_peaks": 100,
                        "ppm": 20,
                        "minimum_tolerance_Da": 0.01,
                        "intensity_weight": "sqrt",
                        "query_OOV": "drop_before_normalization",
                    },
                },
            )
            directory.rename(destination)
        except BaseException:
            shutil.rmtree(directory, ignore_errors=True)
            raise

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text())
        if (
            manifest.get("schema_version") != 1
            or manifest.get("elements") != native.ELEMENTS
            or manifest.get("fingerprint") != "morgan-r2-4096-achiral"
        ):
            raise ValueError("Unsupported sub-formula index schema or element ordering")
        model = cls(
            sparse.load_npz(directory / "train_tokens.npz"),
            np.load(directory / "train_formulas.npy"),
            np.unpackbits(
                np.load(directory / "fingerprints_packed.npy"), axis=1, bitorder="little"
            ),
            np.load(directory / "train_ids.npy", allow_pickle=False),
            json.loads((directory / "vocabulary.json").read_text()),
            manifest["parameters"],
            manifest.get("metadata", {}),
        )
        if manifest["training_rows"] != len(model.ids):
            raise ValueError("Sub-formula index row count differs from its manifest")
        return model

    def _query_features(self, queries):
        matrix, _ = native.build_token_matrix(tokenize(queries), self.vocabulary)
        return matrix, native.formula_vectors([q["formula"] for q in queries])

    def _formula_score(self, query_formulas, weight):
        # Unique training formulas avoid repeatedly evaluating the same counts.
        values = native.weighted_formula_sim(query_formulas, self.unique_formulas, weight, batch=1)
        return values[:, self.formula_inverse]

    def predict(self, queries, *, batch_size=16):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not self.parameters:
            raise ValueError("The neighbour index has no validation-selected parameters")
        p = self.parameters
        if not 1 <= int(p["k"]) <= len(self.ids) or float(p["T"]) <= 0:
            raise ValueError("Invalid sub-formula consensus parameters")
        matrix, formulas = self._query_features(queries)
        prediction = np.empty((len(queries), 4096), np.float32)
        neighbours = np.empty((len(queries), int(p["k"])), np.int64)
        weights = np.empty(neighbours.shape, np.float32)
        for begin in range(0, len(queries), batch_size):
            end = min(begin + batch_size, len(queries))
            spectral = (matrix[begin:end] @ self.matrix.T).toarray().astype(np.float32)
            scores = spectral + np.float32(p["alpha"]) * self._formula_score(
                formulas[begin:end], p["w"]
            )
            index, score = native.topk_sorted(scores, int(p["k"]))
            weight = native.softmax_weights(score, p["T"], int(p["k"]))
            # Match the upstream driver's float32 matrix-vector export path.
            prediction[begin:end] = np.stack(
                [
                    w @ self.fingerprints[i].astype(np.float32)
                    for w, i in zip(weight, index, strict=True)
                ]
            )
            neighbours[begin:end], weights[begin:end] = index, weight
        return prediction, {
            "neighbours": neighbours,
            "weights": weights,
            "empty_token_rows": int((np.diff(matrix.indptr) == 0).sum()),
        }

    def select(self, validation, *, grid=None, batch_size=8):
        if not validation or any(q.get("split") not in (None, "val") for q in validation):
            raise ValueError("Hyperparameter selection requires validation queries")
        if set(self.ids) & {q["id"] for q in validation}:
            raise ValueError("Training and validation query IDs overlap")
        grid = grid or GRID
        ks = [int(k) for k in grid["k"] if k <= len(self.ids)]
        if not ks:
            raise ValueError("No valid neighbour count in the selection grid")
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
            "objective": "binary Morgan4096 Jaccard at0.5",
            "grid": {key: list(value) for key, value in grid.items()},
            "scores": [{"parameters": list(s), "mean_jaccard": mean[s]} for s in settings],
            "test_used_for_selection": False,
        }


def fit(training, validation, output, *, grid=None, dataset="msg"):
    if Path(output).exists():
        raise FileExistsError(output)
    training_keys = {inchikey14(q.get("target_smiles")) for q in training}
    validation_keys = {inchikey14(q.get("target_smiles")) for q in validation}
    if None in training_keys or None in validation_keys:
        raise ValueError("Training and validation targets must be valid molecules")
    if training_keys & validation_keys:
        raise ValueError("Training and validation share 2D molecular structures")
    model = SubformulaIndex.build(training)
    selection = model.select(validation, grid=grid)
    model.save(output, metadata={"selection": selection, "dataset": dataset})
    return selection
