"""Public workflows using actual fingerprint ranking and subprocess inference."""

import json
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from msms_models import MetricProtocol, RunConfig, evaluate, infer, predict
from msms_models.datasets.download import sha256
from msms_models.io import atomic_json, read_records, write_jsonl
from msms_models.smoke import smoke


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
        model="mist-noaug", dataset="msg", queries=query_path, output=tmp_path / "run",
        assets={
            "posterior": str(tmp_path / "posterior.npy"),
            "posterior_ids": str(tmp_path / "ids.npy"),
            "posterior_manifest": str(tmp_path / "manifest.json"),
            "candidates": str(tmp_path / "candidates.jsonl"),
        },
    )
    return config, queries


def test_python_and_worker_use_same_ranking_and_fixed_universe(cached_retrieval):
    config, queries = cached_retrieval
    in_memory = predict(config, queries)
    assert in_memory[0]["candidates"] == ["CCO", "OCC", "CCN"]
    assert in_memory[1]["candidates"][0] == "CCN"
    report = infer(config)
    assert report["status"] == "complete"
    assert read_records(config.output / "predictions.jsonl") == in_memory
    inputs = json.loads((config.output / "inputs.json").read_text())
    assert inputs
    summary, _ = evaluate(
        queries, in_memory[:1],
        MetricProtocol(task="retrieval", cutoffs=(1,), bootstrap_resamples=0),
    )
    assert summary["metrics"]["hit@1"]["mean"] == 0.5


def test_smoke_runs_real_worker_without_changing_parameters(cached_retrieval, tmp_path):
    config, _ = cached_retrieval
    config = replace(config, parameters={"candidate_ties": "stable"})
    report = smoke(config, tmp_path / "smoke", limit=1)
    assert report["status"] == "passed"
    assert report["query_count"] == 1
    assert report["parameters"] == config.parameters
    assert report["metrics_computed"] is True
    with pytest.raises(FileExistsError):
        smoke(config, tmp_path / "smoke", limit=1)


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
        "assert get_model('frigid').method == 'frigid';"
        "assert not {'torch', 'dgl', 'transformers', 'mist'} & set(sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
