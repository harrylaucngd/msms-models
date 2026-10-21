import pytest

torch = pytest.importorskip("torch")


def peaks(formulas, masses, intensities):
    return {"formula": formulas, "formula_mass_no_adduct": masses, "inten": intensities}


SPECTRA = [
    (
        peaks(["C6H12O6", "C6H10O5", "C5H8O4", "C4H6O3"],
              [180.0634, 162.0528, 132.0423, 102.0317], [1.0, 0.6, 0.3, 0.2]),
        181.0707, "C6H12O6",
    ),
    (peaks(["C2H6O"], [46.0419], [1.0]), 47.0491, "C2H6O"),
]


def encode(spectra, additive_bool_mask):
    from msms_models.methods.mist.vendor.nn_module import FormulaTransformer
    from msms_models.methods.mist.vendor.spectrum_utils import prepare_spectrum_batch

    torch.manual_seed(0)
    model = FormulaTransformer(
        hidden_size=64, peak_attn_layers=2, spectra_dropout=0.0,
        pairwise_featurization=True, num_heads=8, form_embedder="pos-cos",
        adduct_class_num=8, no_diffs=False, peak_type_set="full",
        collision_energy_encoder_type="sinusoidal", instrument_class_num=6,
        additive_bool_mask=additive_bool_mask,
    ).eval()
    batch = prepare_spectrum_batch(
        [s[0] for s in spectra], [s[1] for s in spectra], [s[2] for s in spectra],
        ["[M+H]+"] * len(spectra), [20.0] * len(spectra),
    )
    batch["batch_spec_instrument"] = torch.zeros(len(spectra), dtype=torch.long)
    with torch.inference_mode():
        return model(batch)[batch["batch_peak_type"] == 3]


def test_masked_attention_is_independent_of_batch_padding():
    alone = encode(SPECTRA[1:], additive_bool_mask=False)
    padded = encode(SPECTRA, additive_bool_mask=False)[1:]
    torch.testing.assert_close(padded, alone, rtol=1e-5, atol=1e-5)


def test_additive_bool_mask_reproduces_batch_dependence():
    # Paper Sec. 6.1.1: padding receives +1 instead of -inf, so a short spectrum
    # encoded next to a longer one differs from the same spectrum encoded alone.
    alone = encode(SPECTRA[1:], additive_bool_mask=True)
    padded = encode(SPECTRA, additive_bool_mask=True)[1:]
    assert not torch.allclose(padded, alone, atol=1e-4)
