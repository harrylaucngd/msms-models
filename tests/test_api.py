import json
from types import SimpleNamespace

import pytest

from msms_models import api
from msms_models.config import RunConfig
from msms_models.io import write_jsonl


def test_successful_worker_with_wrong_ids_marks_run_failed(tmp_path, monkeypatch):
    query_file = tmp_path / "queries.jsonl"
    write_jsonl(query_file, [{"id": "q", "formula": "C2H6O"}])
    asset = tmp_path / "asset"
    asset.write_text("fixture")
    config = RunConfig(
        model="mist-noaug",
        dataset="msg",
        queries=query_file,
        output=tmp_path / "out",
        assets={
            k: str(asset)
            for k in ["posterior", "posterior_ids", "posterior_manifest", "candidates"]
        },
    )

    def worker(*args, **kwargs):
        write_jsonl(config.output / "predictions.jsonl", [{"id": "wrong", "candidates": ["CCO"]}])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(api.subprocess, "run", worker)
    with pytest.raises(ValueError, match="outside"):
        api.infer(config)
    assert json.loads((config.output / "run.json").read_text())["status"] == "failed"
