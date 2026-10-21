"""GLMR two-stage retrieval port: tokenizer, checkpoint loading, ranking and training."""

import hashlib
import json
import pickle
import sys
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from msms_models.config import RunConfig  # noqa: E402
from msms_models.data.io import write_jsonl  # noqa: E402
from msms_models.models.retrieval.chemformer_two_stage import data as glmr_data  # noqa: E402
from msms_models.models.retrieval.chemformer_two_stage import inference, train  # noqa: E402
from msms_models.models.retrieval.chemformer_two_stage.model import (  # noqa: E402
    ChemformerDecoder,
    ChemformerEncoder,
    CrossFusion,
    SpectralEncoder,
    build_generative_model,
    load_chemformer_checkpoint,
)
from msms_models.models.retrieval.chemformer_two_stage.tokenizer import (  # noqa: E402
    SmilesTokenizer,
)

TINY = {"vocab_size": 523, "d_model": 16, "num_layers": 1, "num_heads": 2, "d_feedforward": 32,
        "dropout": 0.1, "activation": "gelu", "max_seq_len": 128, "pad_token_idx": 0}
SPEC = {"spec_d_model": 16, "spec_num_layers": 1, "spec_num_heads": 2, "spec_d_feedforward": 32,
        "spec_dropout": 0.1, "activation": "gelu", "pad_token_idx": 0}
POOL = ["CCO", "CCCO", "OCCO", "CC(C)O", "C1CC1O", "COC", "CCN", "c1ccccc1O"]


def test_tokenizer_matches_chemformer_vocabulary():
    tokenizer = SmilesTokenizer()
    vocab = [tokenizer.idx2token[i] for i in range(tokenizer.vocab_size)]
    # SHA-256 of GLMR's built-in vocabulary (json.dumps of the ordered token list).
    digest = hashlib.sha256(json.dumps(vocab).encode()).hexdigest()
    assert digest == "5b83db79c174da544aa5db533d6a39980ed1df9bb9b150961478c3e3c8f69d38"
    ids = tokenizer.encode("C[C@H](Cl)c1ccccc1")
    assert ids[0] == 2 and ids[-1] == 3 and tokenizer.token2idx["C"] == 272
    assert tokenizer.decode(ids) == "C[C@H](Cl)c1ccccc1"
    assert tokenizer.decode([4, 0, 3]) == " "  # GLMR does not skip the mask token
    assert len(tokenizer.encode("C" * 600)) == 512


def _chemformer_checkpoint(path):
    """A tiny checkpoint shaped like Chemformer's, with a pickled Lightning callback."""
    torch.manual_seed(0)
    encoder = ChemformerEncoder(**TINY)
    decoder = ChemformerDecoder(**TINY)
    state = {**encoder.state_dict(), **{k: v for k, v in decoder.state_dict().items()
                                        if not k.startswith(("emb.", "pos_emb"))}}
    module = types.ModuleType("pytorch_lightning.callbacks.model_checkpoint")

    class ModelCheckpoint:
        pass

    ModelCheckpoint.__module__ = module.__name__
    ModelCheckpoint.__qualname__ = "ModelCheckpoint"
    module.ModelCheckpoint = ModelCheckpoint
    hp = dict(TINY)
    names = ["pytorch_lightning", "pytorch_lightning.callbacks", module.__name__]
    saved = {name: sys.modules.get(name) for name in names}
    sys.modules.update({name: types.ModuleType(name) for name in names[:2]})
    sys.modules[module.__name__] = module
    try:
        torch.save({"state_dict": {k: v.half() for k, v in state.items()},
                    "hyper_parameters": hp, "callbacks": {ModelCheckpoint: {"monitor": "x"}}}, path)
    finally:
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def _assets(tmp_path, end_bias=-10.0):
    chemformer = tmp_path / "chemformer.ckpt"
    _chemformer_checkpoint(chemformer)
    torch.manual_seed(1)
    spec = SpectralEncoder.from_config(SPEC)
    torch.save({"spec_encoder_state_dict": spec.state_dict(), "config": {"model": SPEC}},
               tmp_path / "pre.pt")
    hp, state = load_chemformer_checkpoint(chemformer)
    decoder = ChemformerDecoder.from_chemformer(hp, state)
    with torch.no_grad():  # emits "C" until max_gen_len, or ends at once (empty SMILES)
        decoder.token_fc.bias.zero_()
        decoder.token_fc.bias[272] = 5.0
        decoder.token_fc.bias[3] = end_bias
    fusion = CrossFusion.from_config(SPEC)
    torch.save({"cross_fusion_state_dict": fusion.state_dict(),
                "decoder_state_dict": decoder.state_dict(), "config": {"model": SPEC}},
               tmp_path / "gen.pt")
    return tmp_path


@pytest.fixture
def assets(tmp_path):
    return _assets(tmp_path)


def test_chemformer_loader_needs_no_lightning(assets):
    hp, state = load_chemformer_checkpoint(assets / "chemformer.ckpt")
    assert hp["d_model"] == 16 and state["emb.weight"].dtype == torch.float16
    encoder = ChemformerEncoder.from_chemformer(hp, state)
    assert encoder.emb.weight.dtype == torch.float32


def _queries(n=5):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        peaks = 70 if i == 0 else 10 + i  # the first exceeds max_peaks
        mz = np.sort(rng.uniform(50, 300, peaks)).round(4)
        intensity = rng.uniform(0.01, 1, peaks).round(3)
        intensity[:3] = intensity[3]  # equal intensities around the top-k cut
        rows.append({"id": f"q{i}", "formula": "C2H6O",
                     "spectrum": {"mz": mz.tolist(), "intensity": intensity.tolist()}})
    return rows


def _config(assets, **parameters):
    queries = _queries()
    pools = [{"id": q["id"], "candidates": POOL[i % 3:] + POOL[: i % 3]} for i, q in
             enumerate(queries)]
    write_jsonl(assets / "candidates.jsonl", pools)
    return RunConfig(
        "chemformer-two-stage", "msg", assets / "q.jsonl", assets / "out",
        assets={"chemformer_checkpoint": str(assets / "chemformer.ckpt"),
                "pre_retrieval_checkpoint": str(assets / "pre.pt"),
                "generative_checkpoint": str(assets / "gen.pt"),
                "candidates": str(assets / "candidates.jsonl")},
        parameters=parameters,
    ), queries, {p["id"]: p["candidates"] for p in pools}


def _reference(model, query, pool, embeddings, k):
    """GLMR's evaluate_full_collect for one query (batch of one is not used by predict)."""
    record = glmr_data.query_record(query)
    spec = glmr_data.prepare_spectrum(record["mzs"], record["intensities"], 61)
    spectra, mask = glmr_data.batch_spectra([spec], "cpu")
    es = model.encode_spectra_pooled(spectra, mask).numpy()[0]
    sims = np.stack([embeddings[s] for s in pool]) @ es
    ranking = np.argsort(-sims)
    return [pool[i] for i in ranking[:k]], [pool[i] for i in ranking[k:]]


@pytest.mark.parametrize("end_bias", [-10.0, 10.0])
def test_predict_reranks_top_k_and_keeps_stage_one_tail(tmp_path, end_bias):
    assets = _assets(tmp_path, end_bias)
    config, queries, pools = _config(assets, pre_retrieval_k=3, eval_batch_size=2, max_gen_len=20)
    rows = inference.predict(config, queries)
    summary = json.loads((assets / "out" / "chemformer_two_stage.json").read_text())
    model = build_generative_model(assets / "chemformer.ckpt", assets / "pre.pt", assets / "gen.pt")
    model.eval()
    tokenizer = SmilesTokenizer()
    embeddings = inference.encode_candidates(model, POOL, tokenizer, torch.device("cpu"))
    for row, query in zip(rows, queries, strict=True):
        pool = pools[query["id"]]
        assert row["id"] == query["id"] and sorted(row["candidates"]) == sorted(pool)
        head, tail = _reference(model, query, pool, embeddings, 3)
        assert row["candidates"][3:] == tail
        generated = summary["generated_smiles"][query["id"]]
        if row["reranked"]:
            gen = inference.encode_candidates(model, [generated], tokenizer, torch.device("cpu"))
            sims = np.stack([embeddings[s] for s in head]) @ gen[generated]
            assert row["candidates"][:3] == [head[i] for i in np.argsort(-sims)]
        else:
            assert generated == "" and row["candidates"][:3] == head
    assert summary["generation_failures"] == sum(not r["reranked"] for r in rows)
    assert summary["generation_failures"] == (0 if end_bias < 0 else len(rows))


def test_prepare_spectrum_keeps_most_intense_peaks_in_mz_order():
    mz = np.arange(1, 71, dtype=np.float32)
    intensity = np.linspace(0, 1, 70, dtype=np.float32)
    spec = glmr_data.prepare_spectrum(mz, intensity, 61)
    assert spec.shape == (61, 2) and spec.dtype == np.float32
    assert np.all(np.diff(spec[:, 0]) > 0) and spec[0, 0] == 10


def test_validate_rejects_bad_parameters(assets):
    config, queries, _ = _config(assets, beam_size=0)
    with pytest.raises(ValueError, match="beam_size"):
        inference.validate(config, queries)
    config, _, _ = _config(assets)
    with pytest.raises(ValueError, match="peaks"):
        inference.validate(config, [{"id": "x", "spectrum": {"mz": [], "intensity": []}}])


# Configurations stored in the released checkpoints, without paths.
STORED_MODEL = {"d_model": 512, "vocab_size": 523, "enc_num_layers": 6, "num_heads": 8,
                "d_feedforward": 2048, "dropout": 0.1, "activation": "gelu", "max_seq_len": 512,
                "pad_token_idx": 0, "spec_d_model": 512, "spec_num_layers": 6,
                "spec_num_heads": 8, "spec_d_feedforward": 2048, "spec_dropout": 0.1}
STORED_PRE = {
    "data": {"max_peaks": 61, "max_smiles_len": 512, "noise_scale": 0.3},
    "model": {**STORED_MODEL, "temperature": 0.1},
    "training": {"epochs": 300, "batch_size": 256, "lr": 0.0001, "weight_decay": 0.1,
                 "num_workers": 4, "val_every_n_epochs": 5, "seed": 42},
    "eval": {"top_k": [1, 5, 20], "mces_threshold": 15},
}
STORED_GEN_EVAL = {"beam_size": 5, "max_gen_len": 512, "top_k": [1, 5, 20],
                   "pre_retrieval_k": 40, "mces_threshold": 15}


@pytest.mark.parametrize("variant", train.VARIANTS)
def test_default_configs_equal_released_checkpoint_configs(variant):
    assert train.logic(train.default_config("pre-retrieval", variant)) == STORED_PRE
    training = {"epochs": 30, "batch_size": 128, "lr": 0.0001, "weight_decay": 0.1,
                "num_workers": 4, "val_every_n_epochs": 5, "seed": 42}
    if variant == "inflated":
        training |= {"accumulation_steps": 4, "val_every_n_epochs": 1}
    expected = {"data": {"max_peaks": 61, "max_smiles_len": 512, "top_k": 40},
                "model": STORED_MODEL, "training": training, "eval": STORED_GEN_EVAL}
    config = train.default_config("generative", variant)
    assert train.logic(config) == expected
    assert config["training"].get("legacy_scaled_clip", False) == (variant == "corrected")


def _mgf(path, rows):
    lines = []
    for name, smiles, fold, peaks in rows:
        lines += ["BEGIN IONS", f"NAME={name}", f"SMILES={smiles}", f"FOLD={fold}"]
        lines += [f"{mz} {intensity}" for mz, intensity in peaks]
        lines += ["END IONS", ""]
    path.write_text("\n".join(lines))


def test_training_smoke_runs_all_stages(assets):
    peaks = [(50.0 + i, 0.1 * (i + 1)) for i in range(6)]
    rows = [(f"s{i}", POOL[i % 4], "train" if i < 4 else "val", peaks) for i in range(6)]
    _mgf(assets / "tiny.mgf", rows)
    with open(assets / "pools.pkl", "wb") as handle:
        pickle.dump({s: POOL for s in POOL[:4]}, handle)
    pre = train.default_config("pre-retrieval", "corrected", mgf=str(assets / "tiny.mgf"),
                               candidates=str(assets / "pools.pkl"),
                               chemformer=str(assets / "chemformer.ckpt"),
                               save_dir=str(assets / "pre"))
    pre["model"] |= SPEC
    pre["training"] |= {"epochs": 1, "batch_size": 2, "num_workers": 0}
    train.train_pre_retrieval(pre, device="cpu", log=lambda *_: None)
    stored = torch.load(assets / "pre" / "best_model.pt", weights_only=False)
    assert stored["config"] == pre and "spec_encoder_state_dict" in stored
    written = train.precompute_candidates(pre, assets / "pre" / "best_model.pt",
                                          assets / "precomputed", top_k=3, device="cpu",
                                          log=lambda *_: None)
    with open(written["train"], "rb") as handle:
        top = pickle.load(handle)
    assert sorted(top) == [0, 1, 2, 3] and all(len(v) == 3 for v in top.values())
    for variant in train.VARIANTS:
        gen = train.default_config(
            "generative", variant, mgf=str(assets / "tiny.mgf"),
            candidates=str(assets / "pools.pkl"), chemformer=str(assets / "chemformer.ckpt"),
            pre_retrieval=str(assets / "pre" / "best_model.pt"),
            precomputed_dir=str(assets / "precomputed"), save_dir=str(assets / variant))
        gen["data"]["train_candidates_path"] = str(written["train"])
        gen["data"]["val_candidates_path"] = str(written["val"])
        gen["model"] |= SPEC
        gen["training"] |= {"epochs": 1, "batch_size": 2, "num_workers": 0}
        loss = train.train_generative(gen, device="cpu", log=lambda *_: None)
        assert np.isfinite(loss)
        stored = torch.load(assets / variant / "best_gen_model.pt", weights_only=False)
        assert set(stored) >= {"cross_fusion_state_dict", "decoder_state_dict", "config"}


# The registry entry this port needs (models/__init__.py is maintained separately).
ENTRY = dict(
    name="chemformer-two-stage", task="retrieval",
    module="msms_models.models.retrieval.chemformer_two_stage.inference",
    formula_required=False,
    assets=("candidates", "chemformer_checkpoint", "pre_retrieval_checkpoint",
            "generative_checkpoint"),
    parameters=("beam_size", "candidate_batch_size", "eval_batch_size", "max_gen_len",
                "max_peaks", "pre_retrieval_k"),
    requires=("torch",),
)


def test_registry_entry_runs_through_the_api(assets, monkeypatch):
    from msms_models import api, models

    entry = models.Model(**ENTRY)
    monkeypatch.setitem(models.MODELS, entry.name, entry)
    config, queries, _ = _config(assets, max_gen_len=20, pre_retrieval_k=40)
    rows = api.predict(config, queries)
    assert [r["id"] for r in rows] == [q["id"] for q in queries]
    assert all(r["reranked"] == len(r["candidates"]) for r in rows)  # pools shorter than K
