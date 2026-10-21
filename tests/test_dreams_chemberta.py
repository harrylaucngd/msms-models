import json
import random
from pathlib import Path

import pytest
import yaml

from msms_models.models.retrieval.dreams_chemberta import candidates, inference, train

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "table3" / "dreams_chemberta"

def upstream_cand_map(library, seed):
    """The loop of SpecBridge's `compute_ranks`, using the global RNG as upstream."""
    state = random.getstate()
    random.seed(seed)
    try:
        cand_map = {}
        for k, vs in library.items():
            vals = [v for v in vs if v]
            if k not in vals:
                vals.append(k)
            random.shuffle(vals)
            cand_map[k] = list(dict.fromkeys(vals))
        return cand_map
    finally:
        random.setstate(state)


def test_frozen_order_matches_upstream_shuffle_across_all_keys():
    library = {f"T{i}": [f"T{i}", *(f"C{i}_{j}" for j in range(i % 7 + 1)), "", f"C{i}_0"]
               for i in range(50)}
    library["missing_true"] = ["A", "B"]
    assert candidates.shuffled_pools(library, 1234) == upstream_cand_map(library, 1234)
    pool = candidates.shuffled_pools(library, 1234)["missing_true"]
    assert sorted(pool) == ["A", "B", "missing_true"]


def test_freeze_candidates_is_id_keyed_and_uses_canonical_lookup(tmp_path):
    pytest.importorskip("rdkit")
    library = {"Oc1ccccc1": ["CCO", "Oc1ccccc1"], "CC": ["CC"]}
    (tmp_path / "pool.json").write_text(json.dumps(library))
    queries = tmp_path / "queries.jsonl"
    queries.write_text(
        json.dumps({"id": "q1", "target_smiles": "c1ccccc1O", "spectrum": {"mz": [1], "intensity": [1]}})
        + "\n"
    )
    out = tmp_path / "frozen.jsonl"
    candidates.freeze_candidates(tmp_path / "pool.json", queries, out, canonical_keys=True)
    row = json.loads(out.read_text())
    assert row["id"] == "q1" and sorted(row["candidates"]) == ["CCO", "Oc1ccccc1"]


def test_candidate_pool_drops_empty_and_duplicate_strings_in_order():
    assert inference.candidate_pool(["B", "", "A", "B", None, "C"]) == ["B", "A", "C"]


@pytest.mark.parametrize("row", ["inflated", "corrected"])
def test_paper_training_configs_differ_only_in_the_molecule_encoder(row):
    config = train.load_config(PAPER / f"train-{row}.yaml")
    assert config["canonical_smiles"] is (row == "corrected")
    assert config["chemberta_from_config"] is (row == "inflated")


def test_paper_inference_configs_name_frozen_inputs():
    for row in ("inflated", "corrected"):
        config = yaml.safe_load((PAPER / f"config-{row}.yaml").read_text())
        assert config["model"] == "dreams-chemberta"
        assert set(config["assets"]) == {"checkpoint", "candidates"}
        assert config["parameters"]["batch_size"] == 32


def test_dreams_backbone_matches_checkpoint_layout_and_ignores_peak_order():
    torch = pytest.importorskip("torch")
    from msms_models.models.retrieval.dreams_chemberta.model import (
        DreamsAdapter,
        DreaMSEncoder,
        ProcrustesResidualMapper,
        pad_peaks,
    )

    torch.manual_seed(0)
    encoder = DreaMSEncoder().eval()
    keys = list(encoder.state_dict())
    assert len(keys) == 65 and "fourier_enc.Fourier frequencies" in keys
    assert encoder.fourier_enc.b.shape == (1, 5997)
    adapter = DreamsAdapter(encoder, d_out=16, hidden=8).eval()
    spectra = [{"mz": [50.0, 120.5, 301.2], "intensity": [0.2, 1.0, 0.5]},
               {"mz": [80.0], "intensity": [1.0]}]
    peaks = pad_peaks(spectra)
    assert peaks.shape == (2, 3, 2) and peaks[1, 1:].abs().sum() == 0
    with torch.no_grad():
        z = adapter(peaks)
        swapped = adapter(peaks[:, [2, 0, 1]][:1])
    assert torch.allclose(z.norm(dim=-1), torch.ones(2), atol=1e-5)
    assert torch.allclose(z[:1], swapped, atol=1e-5)
    mapper = ProcrustesResidualMapper(16, 12, n_blocks=2, hidden=8)
    assert torch.allclose(mapper(z), mapper.W(z))  # zero-initialised residual branches
    adapter.unfreeze_last(2)
    trainable = {n for n, p in adapter.named_parameters() if p.requires_grad}
    assert any(n.startswith("dreams.transformer_encoder.atts.6") for n in trainable)
    assert not any(n.startswith("dreams.transformer_encoder.atts.4") for n in trainable)


def test_predict_ranks_by_dot_product_without_reading_targets(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from msms_models.config import RunConfig
    from msms_models.models.retrieval.dreams_chemberta import model as model_module

    vectors = {"A": [1.0, 0.0], "B": [0.0, 1.0], "C": [0.6, 0.6]}

    class Fake:
        batches = []

        def embed_smiles(self, smiles, device):
            return torch.tensor([vectors[s] for s in smiles])

        def embed_spectra(self, peaks):
            self.batches.append(peaks.shape[0])
            mu = torch.stack([peaks[:, 0, 0] / 100, 1 - peaks[:, 0, 0] / 100], dim=-1)
            return None, mu

    fake = Fake()
    monkeypatch.setattr(model_module, "build_model", lambda *a, **k: (fake, {}))
    (tmp_path / "ckpt.pt").write_bytes(b"")
    (tmp_path / "c.jsonl").write_text(
        "\n".join(json.dumps({"id": i, "candidates": ["A", "B", "C", "A", ""]})
                  for i in ("q1", "q2", "q3"))
    )
    queries = [
        {"id": "q1", "spectrum": {"mz": [90.0], "intensity": [1.0]}},
        {"id": "q2", "spectrum": {"mz": [10.0], "intensity": [1.0]}},
        {"id": "q3", "spectrum": {"mz": [55.0], "intensity": [1.0]}},
    ]
    config = RunConfig(
        model="mist", dataset="msg", queries=tmp_path / "q.jsonl", output=tmp_path / "out",
        assets={"checkpoint": str(tmp_path / "ckpt.pt"), "candidates": str(tmp_path / "c.jsonl")},
        parameters={"batch_size": 2},
    )
    rows = inference.predict(config, queries)
    assert [r["candidates"][0] for r in rows] == ["A", "B", "C"]
    assert all(sorted(r["candidates"]) == ["A", "B", "C"] for r in rows)
    assert rows[0]["scores"] == sorted(rows[0]["scores"], reverse=True)
    assert fake.batches == [2, 1]
