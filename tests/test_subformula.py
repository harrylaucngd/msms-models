import shutil

import numpy as np
import pytest

from msms_models.retrieval.subformula import SubformulaIndex


def query(identifier, smiles, formula, precursor):
    return {
        "id": identifier,
        "target_smiles": smiles,
        "formula": formula,
        "precursor_formula": precursor,
        "spectrum": {"mz": [31.01784, 45.03349], "intensity": [0.5, 1.0]},
    }


def training():
    return [query("a", "CCO", "C2H6O", "C2H7O"), query("b", "CCCO", "C3H8O", "C3H9O")]


def test_prediction_does_not_read_target_smiles_and_accepts_unseen_formula():
    index = SubformulaIndex.build(training())
    index.parameters = {"w": 2.0, "alpha": 2.0, "T": 0.1, "k": 2}
    q = query("q", None, "C4H10O", "C4H11O")
    a, details = index.predict([q])
    q["target_smiles"] = "not a molecule"
    b, _ = index.predict([q])
    assert np.array_equal(a, b)
    assert a.shape == (1, 4096) and np.isfinite(a).all()
    assert details["neighbours"].shape == (1, 2)
    assert np.isclose(details["weights"].sum(), 1.0)


def test_training_index_remains_usable_after_relocation(tmp_path):
    index = SubformulaIndex.build(training())
    index.parameters = {"w": 2.0, "alpha": 2.0, "T": 0.1, "k": 2}
    source, moved = tmp_path / "source", tmp_path / "moved"
    index.save(source, metadata={"dataset": "msg"})
    shutil.move(source, moved)
    loaded = SubformulaIndex.load(moved)
    q = query("q", None, "C4H10O", "C4H11O")
    assert np.array_equal(index.predict([q])[0], loaded.predict([q])[0])


def test_validation_selection_rejects_test_inputs_and_overlapping_ids():
    index = SubformulaIndex.build(training())
    q = query("q", "CCCCO", "C4H10O", "C4H11O")
    q["split"] = "test"
    with pytest.raises(ValueError, match="validation"):
        index.select([q])
    q["split"], q["id"] = "val", "a"
    with pytest.raises(ValueError, match="overlap"):
        index.select([q])


def test_selection_on_a_small_library_uses_valid_neighbour_counts():
    index = SubformulaIndex.build(training())
    q = query("q", "CCCCO", "C4H10O", "C4H11O")
    result = index.select(
        [q], grid={"w": [1.0, 2.0], "alpha": [0.0, 1.0], "T": [0.05, 0.1], "k": [1, 2, 50]}
    )
    assert result["selected"]["k"] in {1, 2}
    assert result["test_used_for_selection"] is False
