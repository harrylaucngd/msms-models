"""Public workflows using actual fingerprint ranking."""

import subprocess
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from msms_models import MetricProtocol, RunConfig, evaluate, predict, run
from msms_models.data.io import read_records, write_jsonl


@pytest.fixture
def cached_retrieval(tmp_path):
    # Posterior order differs from query order. Candidate aliases are retained.
    queries = [
        {"id": "ethanol", "formula": "C2H6O", "target_smiles": "CCO"},
        {"id": "amine", "formula": "C2H7N", "target_smiles": "CCN"},
    ]
    query_path = tmp_path / "queries.jsonl"
    write_jsonl(query_path, queries)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)
    np.save(
        tmp_path / "posterior.npy",
        np.stack([
            generator.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in ("CCN", "CCO")
        ]).astype(np.float32),
    )
    np.save(tmp_path / "ids.npy", np.array(["amine", "ethanol"]))
    write_jsonl(
        tmp_path / "candidates.jsonl",
        [{"id": q["id"], "candidates": ["CCN", "CCO", "OCC"]} for q in queries],
    )
    config = RunConfig(
        model="mist", dataset="msg", queries=query_path, output=tmp_path / "run",
        assets={
            "posterior": str(tmp_path / "posterior.npy"),
            "posterior_ids": str(tmp_path / "ids.npy"),
            "candidates": str(tmp_path / "candidates.jsonl"),
        },
    )
    return config, queries


def test_run_records_the_same_ranking_as_predict(cached_retrieval):
    config, queries = cached_retrieval
    in_memory = predict(config, queries)
    assert in_memory[0]["candidates"] == ["CCO", "OCC", "CCN"]
    assert in_memory[1]["candidates"][0] == "CCN"
    assert run(config)["queries"] == 2
    assert read_records(config.output / "predictions.jsonl") == in_memory
    summary, _ = evaluate(
        queries, in_memory[:1],
        MetricProtocol(task="retrieval", cutoffs=(1,), bootstrap_resamples=0),
    )
    assert summary["metrics"]["hit@1"]["mean"] == 0.5


def test_limited_run_predicts_the_first_queries(cached_retrieval):
    config, _ = cached_retrieval
    assert run(config, limit=1)["queries"] == 1
    assert len(read_records(config.output / "predictions.jsonl")) == 1


def test_model_discovery_does_not_import_neural_runtimes():
    code = (
        "import sys; from msms_models import get_model;"
        "assert get_model('frigid').task == 'de-novo';"
        "assert not {'torch', 'dgl', 'transformers', 'mist'} & set(sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
