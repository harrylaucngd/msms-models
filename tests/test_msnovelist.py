import os
import pickle
import sqlite3
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from msms_models.config import RunConfig
from msms_models.models.de_novo.msnovelist.data import FoldDataset, decode_formula
from msms_models.models.de_novo.msnovelist.inference import (
    MODEL_CONFIG,
    formula_vector,
    rank_candidates,
)


def test_imports_do_not_load_runtime_or_read_working_directory(tmp_path):
    root = str(Path(__file__).resolve().parents[1] / 'src')
    subprocess.run([sys.executable, '-c',
                    'import sys; sys.path.insert(0, sys.argv[1]); '
                    'from msms_models.models.de_novo.msnovelist import inference, train; '
                    'assert "torch" not in sys.modules; assert "jpype" not in sys.modules',
                    root], cwd=tmp_path, check=True)


def test_formula_order_and_legacy_counter():
    expected = [2, 0, 0, 1, 0, 1, 0, 1, 0, 4]
    np.testing.assert_array_equal(formula_vector('C2H4OClBr'), expected)
    counts = Counter(C=2, H=4, O=1, Cl=1, Br=1)
    np.testing.assert_array_equal(decode_formula(pickle.dumps(counts)), expected)


@pytest.mark.parametrize('formula', ['', 'C2H5Na', 'C2H6O+', 'C0H2', '(CH2)2', 'C2garbage'])
def test_invalid_formula_fails(formula):
    with pytest.raises(ValueError):
        formula_vector(formula)


def test_formula_pickle_rejects_non_counter_globals():
    with pytest.raises(pickle.UnpicklingError):
        decode_formula(pickle.dumps(Path('/tmp/untrusted')))
    with pytest.raises(ValueError):
        decode_formula(pickle.dumps({'Na': 1}))


def test_rank_valid_unique_top_scores_without_formula_filter():
    candidates, scores = rank_candidates(
        ['not-smiles', 'CCO', 'OCC', 'COC', 'CCCC', '', 'CC', 'N'],
        [10, -3, -1, -2, -2, 4, float('-inf'), float('nan')],
    )
    assert candidates == ['CCO', 'COC', 'CCCC']
    assert scores == [-1, -2, -2]
    assert rank_candidates(['CCO', 'COC'], [-1, -2], 1) == (['CCO'], [-1])
    assert rank_candidates(['bad'], [0]) == ([], [])
    with pytest.raises(ValueError):
        rank_candidates(['CCO'], [])


def test_connectivity_deduplication_ignores_stereochemistry():
    candidates, scores = rank_candidates(['C[C@H](O)F', 'C[C@@H](O)F'], [-2, -1])
    assert len(candidates) == 1
    assert scores == [-1]


@pytest.fixture
def database(tmp_path):
    path = tmp_path / 'train.db'
    with sqlite3.connect(path) as conn:
        conn.execute('CREATE TABLE compounds '
                     '(id INTEGER PRIMARY KEY, fp BLOB, smiles TEXT, mf BLOB, '
                     'grp TEXT, perm_order INTEGER)')
        for i, fold in enumerate(['fold0', 'fold1', 'fold2']):
            conn.execute('INSERT INTO compounds VALUES (?, ?, ?, ?, ?, ?)',
                         (i, np.zeros(4096, np.uint8).tobytes(), 'CCO',
                          pickle.dumps(Counter(C=2, H=6, O=1)), fold, 0))
    return path


def test_dataset_fold_selection_and_tokenization(database):
    pytest.importorskip('torch')
    before = database.read_bytes()
    train = FoldDataset(database)
    validation = FoldDataset(database, validation=True)
    assert train.ids == [1, 2]
    assert validation.ids == [0]
    fp, formula, tokens, labels = train[0]
    assert fp.shape == (4096,)
    np.testing.assert_array_equal(formula, formula_vector('C2H6O'))
    assert tokens.shape == (127, 39)
    assert labels.shape == (127,)
    assert labels.dtype == np.int64
    from msms_models.models.de_novo.msnovelist.vendor import tokens_process as tokenizer

    assert tokenizer.idx2smiles(np.r_[tokens[0].argmax(), labels]) == 'CCO'
    with train.connect() as conn, pytest.raises(sqlite3.OperationalError):
        conn.execute('DELETE FROM compounds')
    assert database.read_bytes() == before


def test_dataset_schema_missing_file_and_empty_fold(tmp_path, database):
    missing = tmp_path / 'missing.db'
    with pytest.raises(sqlite3.OperationalError):
        FoldDataset(missing)
    assert not missing.exists()
    with pytest.raises(ValueError, match='empty'):
        FoldDataset(database, 9, validation=True)
    with pytest.raises(ValueError, match='fold'):
        FoldDataset(database, 10)
    with sqlite3.connect(database) as conn:
        conn.execute('ALTER TABLE compounds RENAME COLUMN fp TO wrong')
    with pytest.raises(ValueError, match='columns'):
        FoldDataset(database)


def test_dataset_rejects_wrong_fingerprint_and_unknown_tokens(database):
    pytest.importorskip('torch')
    with sqlite3.connect(database) as conn:
        conn.execute('UPDATE compounds SET fp=? WHERE id=1', (bytes(512),))
        conn.execute('UPDATE compounds SET smiles=? WHERE id=2', ('[Na+]',))
    data = FoldDataset(database)
    with pytest.raises(ValueError, match='4096'):
        data[0]
    with pytest.raises(ValueError, match='vocabulary'):
        data[1]


def test_predict_alignment_batches_and_empty_candidates(tmp_path, monkeypatch):
    pytest.importorskip('torch')
    from msms_models.models.de_novo.msnovelist import inference

    ids = ['b', 'a', 'c']
    values = np.zeros((3, 4096), np.float32)
    values[:, 0] = [0.2, 0.1, 0.3]
    np.save(tmp_path / 'fp.npy', values)
    np.save(tmp_path / 'ids.npy', ids)
    observed = []

    class Decoder:
        def __init__(self, checkpoint, **kwargs):
            assert kwargs == {'device': 'cpu', 'beam_size': 100}

        def generate(self, fp, formula):
            observed.append(fp[:, 0].tolist())
            return [(['bad'], [0]) if v > 0.25 else (['CCO', 'OCC'], [-2, -1])
                    for v in fp[:, 0]]

    monkeypatch.setattr(inference, 'MSNovelistDecoder', Decoder)
    config = RunConfig('msnovelist', 'test', tmp_path / 'queries', tmp_path / 'output',
                       assets={'posterior': str(tmp_path / 'fp.npy'),
                               'posterior_ids': str(tmp_path / 'ids.npy'),
                               'decoder_checkpoint': 'unused.pth'},
                       parameters={'batch_size': 2})
    queries = [{'id': i, 'formula': 'C2H6O'} for i in ['a', 'b', 'c']]
    result = inference.predict(config, queries)
    assert [r['id'] for r in result] == ['a', 'b', 'c']
    np.testing.assert_allclose(observed[0], [0.1, 0.2])
    np.testing.assert_allclose(observed[1], [0.3])
    assert result[0]['scores'] == [-1]
    assert result[2]['candidates'] == []
    assert inference.predict(config, []) == []
    np.save(tmp_path / 'fp.npy', np.zeros((3, 3609)))
    with pytest.raises(ValueError, match='4096'):
        inference.predict(config, queries)


def test_cpu_construction_buffers_and_strict_state_roundtrip():
    torch = pytest.importorskip('torch')
    from msms_models.models.de_novo.msnovelist.vendor.model_msnovelist import (
        DecoderModel,
        EncoderModel,
        TranscoderModel,
    )

    torch.set_num_threads(1)
    model = TranscoderModel(config=MODEL_CONFIG)
    state = model.state_dict()
    assert not any('counter_matrix' in k or 'pad_term_mask' in k for k in state)
    for cls in (EncoderModel, DecoderModel):
        target = cls(config=MODEL_CONFIG)
        target.load_state_dict(state, strict=True)
        assert all(b.device.type == 'cpu' for b in target.buffers())
        target.to('meta')
        assert all(b.device.type == 'meta' for b in target.buffers())
    # Constructor/state checks only: do not execute the molecular model.


def test_reference_checkpoint_loading_if_supplied():
    torch = pytest.importorskip('torch')
    path = os.environ.get('MSNOVELIST_REFERENCE_CHECKPOINT')
    if not path:
        pytest.skip('External checkpoint not supplied')
    from msms_models.models.de_novo.msnovelist.inference import MSNovelistDecoder
    from msms_models.models.de_novo.msnovelist.vendor.model_msnovelist import TranscoderModel

    torch.set_num_threads(1)
    MSNovelistDecoder(path)
    TranscoderModel(config=MODEL_CONFIG).load_state_dict(
        torch.load(path, map_location='cpu', weights_only=True), strict=True
    )


def test_training_loss_and_validation_contract_with_toy_model():
    torch = pytest.importorskip('torch')
    from msms_models.models.de_novo.msnovelist.train import run_epoch

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logit = torch.nn.Parameter(torch.zeros(()))

        def forward(self, batch):
            logits = torch.stack([self.logit, -self.logit]).reshape(1, 1, 2).expand(2, 1, 2)
            return logits, torch.zeros(2, 1)

    model = Toy()
    batch = (torch.zeros(2, 4096), torch.ones(2, 10),
             torch.zeros(2, 1, 2), torch.zeros(2, 1, dtype=torch.long))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    value = run_epoch(model, [batch], torch.device('cpu'), optimizer)
    assert value == pytest.approx(np.log(2) + 0.03)
    assert model.logit.item() > 0
    before = model.logit.detach().clone()
    run_epoch(model, [batch], torch.device('cpu'))
    torch.testing.assert_close(before, model.logit)
    assert not model.training
    with pytest.raises(ValueError, match='Empty'):
        run_epoch(model, [], torch.device('cpu'))


def test_training_rejects_singleton_batch_before_model_construction(database, tmp_path):
    pytest.importorskip('torch')
    from msms_models.models.de_novo.msnovelist.train import train

    args = SimpleNamespace(database=str(database), output=str(tmp_path / 'output'), fold=0,
                           epochs=1, batch_size=2, workers=0, seed=42, device='cpu')
    with sqlite3.connect(database) as conn:
        conn.execute("UPDATE compounds SET grp='fold0' WHERE id=2")
    with pytest.raises(ValueError, match='singleton'):
        train(args)
    assert not (tmp_path / 'output').exists()


def test_training_saves_config_and_states_without_overwriting(database, tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    from msms_models.models.de_novo.msnovelist import train as module
    from msms_models.models.de_novo.msnovelist.vendor import model_msnovelist

    monkeypatch.setattr(model_msnovelist, 'TranscoderModel',
                        lambda **kwargs: torch.nn.Linear(2, 1))
    calls = []

    def fake_epoch(model, loader, device, optimizer=None):
        calls.append(optimizer is not None)
        return 0.25 if optimizer is not None else 0.125

    monkeypatch.setattr(module, 'run_epoch', fake_epoch)
    output = tmp_path / 'output'
    args = SimpleNamespace(database=str(database), output=str(output), fold=0,
                           epochs=2, batch_size=2, workers=0, seed=42, device='cpu')
    module.train(args)
    assert calls == [True, False, True, False]
    import json

    config = json.loads((output / 'config.json').read_text())
    assert config['seed'] == 42 and config['fingerprint_simulation'] is False
    assert config['model_config'] == MODEL_CONFIG
    weights = list(output.glob('w-*.pth'))
    assert len(weights) == 2
    assert len(list(output.glob('optimizer-*.pth'))) == 2
    assert set(torch.load(weights[0], weights_only=True)) == {'weight', 'bias'}
    with pytest.raises(FileExistsError):
        module.train(args)
    assert calls == [True, False, True, False]
