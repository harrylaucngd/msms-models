import numpy as np
import pytest

from msms_models.models.de_novo.ms_gpt.inference import formula_vector, rank_generated
from msms_models.models.de_novo.ms_gpt.vendor.queries import density_calibrated_query_band


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

    from msms_models.evaluation.chemistry import inchikey14

    encoded = safe.SAFEConverter(slicer=None, ignore_stereo=True).encoder("CCO", allow_empty=True)
    assert inchikey14(safe.decode(encoded)) == inchikey14("CCO")


def _small_model():
    from msms_models.models.de_novo.ms_gpt.vendor.model import ConditionalMoleculeLanguageModel

    return ConditionalMoleculeLanguageModel(
        vocab_size=98, d_model=32, n_layers=2, n_heads=4, fp_bits=4096, fp_encoder_layers=1,
        cross_attn_every_n=1, max_seq_len=16, dropout=0.0,
    ).eval()


def test_cached_generation_matches_full_prefix_greedy_decoding():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = _small_model()
    fp = torch.zeros(1, 4096)
    fp[:, [3, 12, 45]] = 1
    formula = torch.zeros(1, 14)
    formula[:, 0] = 2
    with torch.inference_mode():
        actual = model.generate(
            fp, formula, max_len=8, top_k=1, top_p=1, bos_token_id=0, eos_token_id=97
        )
        expected = torch.zeros(1, 1, dtype=torch.long)
        for _ in range(7):
            next_token = model(expected, fp, formula).logits[:, -1].argmax(-1, keepdim=True)
            expected = torch.cat((expected, next_token), dim=1)
            if int(next_token) == 97:
                break
    torch.testing.assert_close(actual, expected)

