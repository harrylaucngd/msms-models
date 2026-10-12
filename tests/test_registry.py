import importlib
from pathlib import Path

from msms_models.config import RunConfig
from msms_models.registry import MODELS
from msms_models.validation import required_assets


def test_registered_backends_are_importable_without_loading_models():
    for model in MODELS.values():
        module = importlib.import_module("msms_models.backends." + model.backend)
        assert callable(module.predict)


def test_example_configs_cover_models_and_required_assets():
    root = Path(__file__).resolve().parents[1]
    for name in MODELS:
        config = RunConfig.from_file(root / "configs" / f"{name}.yaml")
        assert config.model == name
        assert set(required_assets(config)) <= set(config.assets)


def test_unknown_method_parameter_is_not_silently_ignored(tmp_path):
    import pytest

    from msms_models.validation import validate_run

    config = RunConfig(
        model="mist-noaug",
        dataset="msg",
        queries=tmp_path / "q.jsonl",
        output=tmp_path / "out",
        parameters={"fingerprint_representaton": "binary"},
    )
    with pytest.raises(ValueError, match="Unknown mist-noaug parameters"):
        validate_run(config, [{"id": "q", "formula": "C2H6O"}])
