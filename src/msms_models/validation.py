"""Validate portable run assets before importing a model implementation."""

from __future__ import annotations

import json
import subprocess

from .registry import MODELS, get_model


def required_assets(config):
    return get_model(config.model).required_assets(config)


def validate_run(config, queries):
    model = get_model(config.model)
    if config.dataset not in model.datasets:
        raise ValueError(f"{model.name} does not support dataset {config.dataset!r}")
    if config.workers < 1:
        raise ValueError("workers must be at least one")
    unknown = set(config.parameters) - PARAMETERS[config.model]
    if unknown:
        raise ValueError(f"Unknown {config.model} parameters: {sorted(unknown)}")
    if model.formula_required and any(not q.get("formula") for q in queries):
        raise ValueError(f"{model.name} requires a known molecular formula for every query")
    model.validate(config, queries)
    config.require_assets(*required_assets(config))
    if not isinstance(config.asset_sha256, dict):
        raise ValueError("asset_sha256 must map asset names to SHA-256 hashes")
    if config.asset_sha256:
        from .datasets.download import sha256

        for name, expected in config.asset_sha256.items():
            path = config.require_assets(name)[name]
            if not path.is_file():
                raise ValueError(f"asset_sha256 requires a file, not a directory: {name}")
            if not isinstance(expected, str) or sha256(path) != expected:
                raise ValueError(f"Asset checksum mismatch: {name} ({path})")


def doctor(config, queries):
    validate_run(config, queries)
    modules = get_model(config.model).required_modules(config)
    code = "import importlib.util,json,sys;print(json.dumps({m:importlib.util.find_spec(m) is not None for m in json.loads(sys.argv[1])}))"
    result = subprocess.run(
        [config.python, "-c", code, json.dumps(sorted(set(modules)))],
        capture_output=True,
        text=True,
        check=True,
    )
    dependencies = json.loads(result.stdout)
    return {
        "model": config.model,
        "dataset": config.dataset,
        "queries": len(queries),
        "assets_present": required_assets(config),
        "python": config.python,
        "dependencies": dependencies,
        "ready_for_checkpoint_test": all(dependencies.values()),
        "checkpoint_compatibility_checked": False,
    }



PARAMETERS = {name: set(model.parameters) for name, model in MODELS.items()}
