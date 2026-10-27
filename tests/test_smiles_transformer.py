import json
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tokenizers")
pytest.importorskip("selfies")

from msms_models.config import RunConfig  # noqa: E402
from msms_models.models.de_novo.smiles_transformer import inference  # noqa: E402
from msms_models.models.de_novo.smiles_transformer.model import SmilesTransformer  # noqa: E402
from msms_models.models.de_novo.smiles_transformer.tokenizers import (  # noqa: E402
    EOS,
    PAD,
    SOS,
    UNK,
    Tokenizer,
)
from msms_models.models.de_novo.smiles_transformer.train import train  # noqa: E402

SMILES = ["CCO", "CCN", "c1ccccc1O", "CC(=O)O", "CCCl", "OCC(O)CO"]


def query(i, smiles, formula):
    return {"id": f"q{i}", "target_smiles": smiles, "formula": formula, "precursor_mz": 47.05,
            "spectrum": {"mz": [15.0, 29.0, 31.0], "intensity": [0.2, 1.0, 0.5]}}


QUERIES = [query(i, s, f) for i, (s, f) in enumerate(zip(
    SMILES, ["C2H6O", "C2H7N", "C6H6O", "C2H4O2", "C2H5Cl", "C3H8O3"]))]


def test_imports_do_not_load_torch():
    root = str(Path(__file__).resolve().parents[1] / "src")
    code = ("import sys; sys.path.insert(0, sys.argv[1]); "
            "from msms_models.models.de_novo.smiles_transformer import inference, train; "
            "assert 'torch' not in sys.modules")
    subprocess.run([sys.executable, "-c", code, root], check=True)


def test_selfies_vocabulary_layout_and_round_trip():
    tokenizer = Tokenizer.train("selfies", SMILES * 2, max_len=20)
    vocab = tokenizer.tokenizer.get_vocab()
    alphabet = sorted(t for t in vocab if t not in {PAD, SOS, EOS, UNK})
    # Alphabet ids 0..n-1 in sorted order, then <unk>, then the special tokens.
    assert [vocab[t] for t in alphabet] == list(range(len(alphabet)))
    assert [vocab[t] for t in (UNK, PAD, SOS, EOS)] == list(range(len(alphabet), len(alphabet) + 4))
    ids = tokenizer.encode_batch(["CCO"])[0]
    assert len(ids) == 20 and ids[0] == vocab[SOS] and ids[-1] == vocab[PAD]
    assert tokenizer.decode_batch([ids]) == ["CCO"]


def test_bpe_pads_truncates_and_keeps_tokens_after_end():
    tokenizer = Tokenizer.train("smiles-bpe", SMILES, molecules=SMILES * 5, max_len=12)
    ids = tokenizer.encode_batch(["CCO", "C" * 40])
    assert [len(i) for i in ids] == [12, 12]
    assert ids[1][-1] == tokenizer.token_to_id(EOS)  # truncated inside the template
    assert tokenizer.decode_batch([ids[0]]) == ["CCO"]
    tail = ids[0][:ids[0].index(tokenizer.token_to_id(EOS)) + 1] + tokenizer.encode_batch(["N"])[0][1:2]
    assert tokenizer.decode_batch([tail]) == ["CCON"]


def test_teacher_forcing_shapes_and_decoding_stop():
    torch.manual_seed(0)
    model = SmilesTransformer(30, pad_id=0, d_model=16, nhead=2, num_encoder_layers=1,
                              num_decoder_layers=1).eval()
    spec = torch.rand(3, 61, 2)
    spec[:, 5:] = 0
    formula = torch.rand(3, 118)
    tokens = torch.randint(1, 30, (3, 9))
    logits, targets = model(spec, formula, tokens)
    assert logits.shape == (3 * 8, 30) and torch.equal(targets, tokens.T[1:].reshape(-1))
    with torch.no_grad():
        model.tgt_decoder.bias.fill_(-1e4)
        model.tgt_decoder.bias[7] = 1e4  # every sequence emits token 7 first
    out = model.decode(spec, formula, sos_id=1, eos_id=7, max_len=20, temperature=None)
    assert out.tolist() == [[1, 7]] * 3


def test_training_and_prediction_round_trip(tmp_path):
    output = tmp_path / "selfies"
    train("selfies", QUERIES, QUERIES[:2], output, d_model=16, nhead=2, num_encoder_layers=1,
          num_decoder_layers=1, batch_size=4, max_epochs=1, max_len=20)
    checkpoint = torch.load(output / "model.pt", weights_only=False)
    assert checkpoint["settings"]["tokenizer"] == "selfies" and (output / "tokenizer.json").exists()
    config = RunConfig("smiles-transformer", "msg", tmp_path / "q.jsonl", tmp_path / "out",
                       assets={"checkpoint": str(output / "model.pt")},
                       parameters={"k_predictions": 3, "batch_size": 4}, seed=1)
    rows = inference.predict(config, QUERIES)
    assert [r["id"] for r in rows] == [q["id"] for q in QUERIES]
    assert all(len(r["candidates"]) == 3 and r["status"] == "ok" for r in rows)
    assert inference.predict(config, QUERIES) == rows  # seeded sampling
    json.dumps(rows)
