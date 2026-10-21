import json

import pytest

from msms_models import api
from msms_models.config import RunConfig
from msms_models.data.io import write_jsonl
from msms_models.models import get_model


def test_wrong_prediction_ids_mark_run_failed(tmp_path, monkeypatch):
    query_file = tmp_path / "queries.jsonl"
    write_jsonl(query_file, [{"id": "q", "formula": "C2H6O"}])
    asset = tmp_path / "asset"
    asset.write_text("fixture")
    config = RunConfig(
        model="mist",
        dataset="msg",
        queries=query_file,
        output=tmp_path / "out",
        assets={
            k: str(asset)
            for k in ["posterior", "posterior_ids", "posterior_manifest", "candidates"]
        },
    )
    module = get_model("mist").implementation()
    monkeypatch.setattr(module, "predict", lambda config, queries: [{"id": "wrong", "candidates": []}])
    with pytest.raises(ValueError, match="outside"):
        api.run(config)
    assert json.loads((config.output / "run.json").read_text())["status"] == "failed"
    assert not (config.output / "predictions.jsonl").exists()
