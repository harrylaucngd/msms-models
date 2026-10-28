import importlib
from pathlib import Path

import yaml

from msms_models.config import RunConfig
from msms_models.models import MODELS

ROOT = Path(__file__).resolve().parents[1]


def example_configs():
    for path in sorted((ROOT / "paper").rglob("config*.yaml")):
        if "model" in yaml.safe_load(path.read_text()):
            yield RunConfig.from_file(path)


def test_registered_models_are_importable_without_loading_runtimes():
    for model in MODELS.values():
        assert callable(importlib.import_module(model.module).predict)


def test_example_configs_cover_every_model():
    configs = list(example_configs())
    assert {c.model for c in configs} == set(MODELS)
    for config in configs:
        if "decoder_config" in config.assets:
            assert Path(config.assets["decoder_config"]).is_file()
