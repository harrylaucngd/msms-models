import json

import pytest

torch = pytest.importorskip("torch")


def tree(formula, ion, fragments):
    return {
        "cand_form": formula,
        "cand_ion": ion,
        "output_tbl": {
            "formula": [f for f, _ in fragments],
            "ms2_inten": [i for _, i in fragments],
            "ions": [ion] * len(fragments),
        },
    }


TREES = [
    tree("C6H12O6", "[M+H]+", [("C6H10O5", 1.0), ("C5H8O4", 0.4), ("C4H6O3", 0.2)]),
    tree("C2H6O", "[M+Na]+", [("C2H4", 1.0)]),
    tree("C3H7NO2", "[M+H]+", []),
]


def encode(trees, padding, batch_size):
    from msms_models.encoders.mist.diffms import encode_batches
    from msms_models.encoders.mist.vendor.featurizers import featurize
    from msms_models.encoders.mist.vendor.spectra_encoder import SpectraEncoderGrowing

    torch.manual_seed(0)
    model = SpectraEncoderGrowing(
        form_embedder="pos-cos", output_size=64, hidden_size=32, spectra_dropout=0.0,
        top_layers=1, refine_layers=2, magma_modulo=16, peak_attn_layers=2, num_heads=4,
        set_pooling="cls", pairwise_featurization=True, embed_instrument=False,
        inten_transform="float", additive_bool_mask=padding == "additive-bool",
    ).eval()
    features = [featurize(t, f"s{i}") for i, t in enumerate(trees)]
    return torch.from_numpy(encode_batches(model, features, batch_size, torch.device("cpu")))


def test_masked_padding_is_batch_invariant():
    alone = encode(TREES, "masked", batch_size=1)
    batched = encode(TREES, "masked", batch_size=3)
    torch.testing.assert_close(batched, alone, rtol=1e-5, atol=1e-6)


def test_additive_bool_padding_depends_on_batch():
    # Paper Sec. 6.1.1: padded positions receive +1 instead of -inf, so the shorter
    # spectra change when encoded next to a longer one; the longest does not.
    alone = encode(TREES, "additive-bool", batch_size=1)
    batched = encode(TREES, "additive-bool", batch_size=3)
    torch.testing.assert_close(batched[0], alone[0], rtol=1e-5, atol=1e-6)
    assert not torch.allclose(batched[1:], alone[1:], atol=1e-4)
    # Without padding the historical mask is a no-op.
    torch.testing.assert_close(alone, encode(TREES, "masked", batch_size=1))


def test_trees_are_read_by_query_id(tmp_path):
    from msms_models.encoders.mist.diffms import load_trees

    bundle = tmp_path / "trees.jsonl"
    bundle.write_text(json.dumps({"id": "a", "subformulae": TREES[0]}) + "\n")
    query = {"id": "a", "formula": "C6H12O6", "adduct": "[M+H]+"}
    assert load_trees(bundle, [query]) == [TREES[0]]


def test_lightning_checkpoints_load_the_spectra_encoder(tmp_path):
    from msms_models.encoders.mist.diffms import build_model, load_state

    model = build_model("massspecgym")
    state = model.state_dict()
    assert all(k.startswith("spectra_encoder.") for k in state)
    torch.save({"state_dict": state | {"fp_head.weight": torch.zeros(1)}}, tmp_path / "mist.ckpt")
    model.load_state_dict(load_state(tmp_path / "mist.ckpt"))
    torch.save(model.state_dict(), tmp_path / "bare.pt")
    assert load_state(tmp_path / "bare.pt").keys() == model.state_dict().keys()
