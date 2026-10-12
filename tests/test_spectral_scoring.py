import numpy as np
import pytest

from msms_models._vendor.spectral_scoring import bin_spectrum, expand_scores_to_full_candidates
from msms_models.backends.simulation import score_candidates, validate_metadata


def test_entropy_disjoint_identical_and_missing():
    a = (np.array([3, 8]), np.array([1.0, 0.3], np.float32))
    b = (np.array([30, 80]), np.array([1.0, 0.3], np.float32))
    scores = score_candidates([a, b, None], a)
    assert scores[0] == pytest.approx(0, abs=1e-6)
    np.testing.assert_allclose(scores[1:], 1, atol=1e-6)


def test_bin_boundaries_and_duplicate_pooling():
    spectrum = np.array([[0, 1], [1, 0.2], [1, 0.5], [1500, 10]], np.float32)
    bins, values = bin_spectrum(spectrum, 150000, 1500, "max")
    np.testing.assert_array_equal(bins, [1, 100])
    np.testing.assert_allclose(values, [1, 0.5])
    bins, values = bin_spectrum(spectrum, 150000, 1500, "add")
    np.testing.assert_allclose(values, [1, 0.7])


def test_alias_expansion_preserves_full_universe():
    scores = expand_scores_to_full_candidates(
        ["AAA-X", "BBB-X"], [0.2, 0.4], ["BBB-Y", "AAA-X", "AAA-X", "CCC-X"]
    )
    np.testing.assert_allclose(scores[:3], [0.4, 0.2, 0.2])
    assert np.isnan(scores[3])


def test_missing_energy_is_not_imputed_from_target():
    with pytest.raises(ValueError, match="collision_energy"):
        validate_metadata({"id": "q", "target_smiles": "CCO"})
