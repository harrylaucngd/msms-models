import importlib
from pathlib import Path

import pytest
import yaml

from msms_models.api import validate
from msms_models.config import RunConfig
from msms_models.models import MODELS

ROOT = Path(__file__).resolve().parents[1]


def example_configs():
    paths = [*(ROOT / "configs").rglob("*.yaml"), *(ROOT / "paper").rglob("config*.yaml")]
    for path in sorted(paths):
        if "model" in yaml.safe_load(path.read_text()) and path.name != "architecture.yaml":
            yield RunConfig.from_file(path)


def test_registered_models_are_importable_without_loading_runtimes():
    for model in MODELS.values():
        assert callable(importlib.import_module(model.module).predict)


def test_example_configs_cover_every_model_and_its_required_assets():
    configs = list(example_configs())
    assert {c.model for c in configs} == set(MODELS)
    for config in configs:
        assert set(MODELS[config.model].required_assets(config)) <= set(config.assets)
        assert set(config.parameters) <= set(MODELS[config.model].parameters)
        if "decoder_config" in config.assets:
            assert Path(config.assets["decoder_config"]).is_file()


def test_unknown_model_parameter_is_not_silently_ignored(tmp_path):
    config = RunConfig(
        model="mist", dataset="msg", queries=tmp_path / "q.jsonl", output=tmp_path / "out",
        parameters={"fingerprint_representaton": "binary"},
    )
    with pytest.raises(ValueError, match="Unknown mist parameters"):
        validate(config, [{"id": "q", "formula": "C2H6O"}])
