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


def test_safe_codec_roundtrip_when_installed():
    import importlib.util

    if importlib.util.find_spec("safe") is None:
        pytest.skip("SAFE is an optional generator dependency")
    # An installed but incompatible SAFE/Transformers pair must fail, not skip.
    import safe

    from msms_models.chemistry import inchikey14

    encoded = safe.SAFEConverter(slicer=None, ignore_stereo=True).encoder("CCO", allow_empty=True)
    assert inchikey14(safe.decode(encoded)) == inchikey14("CCO")
