"""Public workflows using actual fingerprint ranking."""

import json
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from msms_models import MetricProtocol, RunConfig, evaluate, predict, run
from msms_models.data.download import sha256
from msms_models.data.io import atomic_json, read_records, write_jsonl


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
    atomic_json(
        tmp_path / "manifest.json",
        {"dataset": "msg", "fingerprint": "morgan-r2-4096-achiral"},
    )
    write_jsonl(
        tmp_path / "candidates.jsonl",
        [{"id": q["id"], "candidates": ["CCN", "CCO", "OCC"]} for q in queries],
    )
    config = RunConfig(
        model="mist", dataset="msg", queries=query_path, output=tmp_path / "run",
        assets={
            "posterior": str(tmp_path / "posterior.npy"),
            "posterior_ids": str(tmp_path / "ids.npy"),
            "posterior_manifest": str(tmp_path / "manifest.json"),
            "candidates": str(tmp_path / "candidates.jsonl"),
        },
    )
    return config, queries


def test_run_records_the_same_ranking_as_predict(cached_retrieval):
    config, queries = cached_retrieval
    in_memory = predict(config, queries)
    assert in_memory[0]["candidates"] == ["CCO", "OCC", "CCN"]
    assert in_memory[1]["candidates"][0] == "CCN"
    report = run(config)
    assert report["status"] == "complete"
    assert read_records(config.output / "predictions.jsonl") == in_memory
    inputs = json.loads((config.output / "inputs.json").read_text())
    assert inputs
    summary, _ = evaluate(
        queries, in_memory[:1],
        MetricProtocol(task="retrieval", cutoffs=(1,), bootstrap_resamples=0),
    )
    assert summary["metrics"]["hit@1"]["mean"] == 0.5


def test_limited_run_keeps_parameters_and_refuses_overwrite(cached_retrieval):
    config, _ = cached_retrieval
    config = replace(config, parameters={"candidate_ties": "stable"})
    report = run(config, limit=1)
    assert report["status"] == "complete"
    assert report["query_count"] == 1
    assert report["config"]["parameters"] == config.parameters
    assert len(read_records(config.output / "predictions.jsonl")) == 1
    with pytest.raises(FileExistsError):
        run(config, limit=1)


def test_pinned_assets_reject_substitution(cached_retrieval):
    config, queries = cached_retrieval
    digest = sha256(config.assets["posterior"])
    assert predict(replace(config, asset_sha256={"posterior": digest}), queries)
    with pytest.raises(ValueError, match="checksum mismatch"):
        predict(replace(config, asset_sha256={"posterior": "0" * 64}), queries)


def test_python_api_rejects_duplicate_query_ids(cached_retrieval):
    config, queries = cached_retrieval
    with pytest.raises(ValueError, match="Duplicate query"):
        predict(config, [queries[0], queries[0]])


def test_model_discovery_does_not_import_neural_runtimes():
    code = (
        "import sys; from msms_models import get_model;"
        "assert get_model('frigid').task == 'de-novo';"
        "assert not {'torch', 'dgl', 'transformers', 'mist'} & set(sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
