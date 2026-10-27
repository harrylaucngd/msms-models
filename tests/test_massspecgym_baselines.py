import json

import numpy as np
import pytest

from msms_models.config import RunConfig
from msms_models.data.io import write_jsonl
from msms_models.massspecgym.transforms import (
    bin_spectrum,
    formula_vector,
    preprocess_peaks,
    tokenize_spectrum,
)
from msms_models.models.retrieval import random as random_retrieval

QUERY = {"id": "q", "formula": "C4H10O", "precursor_mz": 75.08, "target_smiles": "CCCCO",
         "spectrum": {"mz": [5.0, 29.0, 31.0, 41.0, 56.0, 1003.0],
                      "intensity": [9.0, 2.0, 4.0, 4.0, 1.0, 8.0]}}
CANDIDATES = ["CC(C)CO", "CCCCO", "CCOCC", "CCC(C)O", "CCCC"]


def test_peak_preprocessing_matches_matchms_filters():
    # select_by_mz keeps 10 <= m/z <= mz_to; the most intense peaks are kept in m/z order.
    mz, intensity = preprocess_peaks(QUERY["spectrum"]["mz"], QUERY["spectrum"]["intensity"],
                                     mz_to=1000, n_max_peaks=3)
    assert mz.tolist() == [29.0, 31.0, 41.0]
    np.testing.assert_allclose(intensity, [0.5, 1.0, 1.0])
    mz, _ = preprocess_peaks([10.0, 20.0, 30.0], [1.0, 1.0, 1.0], n_max_peaks=2)
    assert mz.tolist() == [20.0, 30.0]  # NumPy 1.24's argsort keeps the last tied peaks


def test_tokenized_spectrum_leads_with_the_precursor_and_is_zero_padded():
    rows = tokenize_spectrum(QUERY, n_peaks=60, max_mz=1000)
    assert rows.shape == (61, 2)
    assert rows[0].tolist() == [75.08, 1.1]
    assert rows[1:5, 0].tolist() == [29.0, 31.0, 41.0, 56.0]
    assert not rows[5:].any()


def test_binned_spectrum_sums_relative_intensities_per_dalton():
    binned = bin_spectrum(QUERY, max_mz=1005)
    assert binned.shape == (1005,)
    assert binned[1003] == 1.0 and binned[29] == pytest.approx(0.25)
    assert binned[5] == 0  # below m/z 10


def test_formula_vector_counts_hydrogens_in_massspecgym_order():
    vector = formula_vector("C4H10O")
    assert vector[0] == 10 and vector[5] == 4 and vector[7] == 1 and vector.sum() == 15


def test_random_retrieval_is_seeded_and_formula_filter_ranks_matching_candidates_first(tmp_path):
    write_jsonl(tmp_path / "c.jsonl", [{"id": "q", "candidates": CANDIDATES}])
    config = RunConfig("random", "msg", tmp_path / "q.jsonl", tmp_path / "out",
                       assets={"candidates": str(tmp_path / "c.jsonl")}, seed=3)
    first = random_retrieval.predict(config, [QUERY])
    assert first == random_retrieval.predict(config, [QUERY])
    assert sorted(first[0]["candidates"]) == sorted(CANDIDATES)
    config = RunConfig("random", "msg", tmp_path / "q.jsonl", tmp_path / "out",
                       assets={"candidates": str(tmp_path / "c.jsonl")},
                       parameters={"formula_filter": True}, seed=3)
    ranked = random_retrieval.predict(config, [QUERY])[0]["candidates"]
    assert set(ranked[:4]) == {"CC(C)CO", "CCCCO", "CCOCC", "CCC(C)O"} and ranked[4] == "CCCC"


def test_mlp_has_pyg_layout():
    pytest.importorskip("torch")
    from msms_models.massspecgym.layers import mlp

    layers = [type(m).__name__ for m in mlp(2, 8, 4, num_layers=3, dropout=0.1)]
    assert layers == ["Linear", "ReLU", "Dropout", "Linear", "ReLU", "Dropout", "Linear"]


@pytest.mark.parametrize("module, options", [
    ("deepsets", {"fourier_features": True, "hidden_channels": 16, "num_layers_per_mlp": 2}),
    ("deepsets", {"fourier_features": False, "hidden_channels": 16, "num_layers_per_mlp": 2}),
    ("fingerprint_ffn", {"hidden_channels": 16, "num_layers": 2}),
])
def test_training_then_prediction_ranks_every_candidate(tmp_path, module, options):
    pytest.importorskip("torch")
    import importlib

    method = importlib.import_module(f"msms_models.models.retrieval.{module}")
    settings = method.DEFAULTS | options | {"batch_size": 2, "max_epochs": 1}
    from msms_models.massspecgym.retrieval import fit_fingerprints

    queries = [QUERY | {"id": f"q{i}"} for i in range(4)]
    fit_fingerprints(method.build, method.featurize, queries, queries[:2], settings, tmp_path / "ckpt")
    write_jsonl(tmp_path / "c.jsonl", [{"id": "q", "candidates": CANDIDATES}])
    config = RunConfig(module, "msg", tmp_path / "q.jsonl", tmp_path / "out",
                       assets={"checkpoint": str(tmp_path / "ckpt/model.pt"),
                               "candidates": str(tmp_path / "c.jsonl")})
    row = method.predict(config, [QUERY])[0]
    assert sorted(row["candidates"]) == sorted(CANDIDATES)
    assert row["scores"] == sorted(row["scores"], reverse=True)
    assert json.dumps(row)
