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
        for dataset in MODELS[name].datasets:
            path = root / "configs" / name / f"{dataset}.yaml"
            config = RunConfig.from_file(path)
            assert config.model == name and config.dataset == dataset
            assert set(required_assets(config)) <= set(config.assets)
            if "decoder_config" in config.assets:
                assert Path(config.assets["decoder_config"]).is_file()
        assert not (root / "configs" / f"{name}.yaml").exists()


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


def test_paired_ms_gpt_configs_keep_same_query_budget():
    import yaml

    root = Path(__file__).resolve().parents[1]
    for dataset in ("msg", "canopus"):
        configs = [
            RunConfig.from_file(root / "configs" / name / f"{dataset}.yaml")
            for name in ("ms-gpt-base", "ms-gpt")
        ]
        for config in configs:
            assert config.parameters["num_queries"] == 20
            assert config.parameters["samples_per_query"] == 5
            assert (
                yaml.safe_load(Path(config.assets["decoder_config"]).read_text())["tokenizer"]
                == "safe"
            )
        assert "adapter_checkpoint" not in configs[0].assets
        assert "adapter_checkpoint" in configs[1].assets
