import numpy as np
import pytest

from msms_models._vendor.ms_gpt.queries import density_calibrated_query_band
from msms_models.backends.ms_gpt import formula_vector, rank_generated


def test_ms_gpt_threshold_is_strictly_greater():
    result = density_calibrated_query_band(np.array([0.1, 0.3, 0.5], np.float32), 3, 0.1, 0.5)
    np.testing.assert_array_equal(result, [[0, 1, 1], [0, 0, 1], [0, 0, 0]])


def test_ms_gpt_formula_counts_and_unsupported():
    np.testing.assert_array_equal(formula_vector("C2H6O")[:4], [2, 6, 0, 1])
    with pytest.raises(ValueError, match="vocabulary"):
        formula_vector("C2H5NaO")


def test_ms_gpt_ranking_uses_formula_and_frequency_without_target():
    vector = np.zeros(4096, np.float32)
    assert rank_generated(["CCO", "C(C)O", "COC", "CCC"], vector, "C2H6O", (0.01, 0.26)) == [
        "CCO",
        "COC",
    ]


def test_diffms_formula_graph_uses_implicit_hydrogens():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from msms_models.backends.diffms import formula_graph

    graph = formula_graph("C2H6O", ["C", "O", "P", "N", "S", "Cl", "F", "H"], torch.zeros(2048))
    assert graph.num_nodes == 3
    assert graph.x.argmax(1).tolist() == [0, 0, 1]
    assert graph.edge_index.numel() == 0
    with pytest.raises(ValueError, match="Unsupported"):
        formula_graph("C2H5Br", ["C", "O", "P", "N", "S", "Cl", "F", "H"], torch.zeros(2048))
