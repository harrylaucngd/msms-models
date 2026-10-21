"""Validate inputs, run a model in the current interpreter and record the run."""

from __future__ import annotations

import importlib.util
from dataclasses import asdict

from .config import RunConfig
from .data.download import sha256
from .data.io import (
    align_predictions,
    atomic_json,
    load_queries,
    normalize_query,
    record_id,
    write_jsonl,
)
from .models import get_model
from .provenance import environment_record, input_record


def validate(config: RunConfig, queries: list[dict]) -> None:
    """Check the query manifest, parameters and assets without loading a model."""
    model = get_model(config.model)
    identifiers = [q["id"] for q in queries]
    if not identifiers:
        raise ValueError("The query manifest is empty")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Duplicate query ids are not allowed")
    if config.dataset not in model.datasets:
        raise ValueError(f"{model.name} does not support dataset {config.dataset!r}")
    unknown = set(config.parameters) - set(model.parameters)
    if unknown:
        raise ValueError(f"Unknown {model.name} parameters: {sorted(unknown)}")
    if model.formula_required and any(not q.get("formula") for q in queries):
        raise ValueError(f"{model.name} requires a known molecular formula for every query")
    config.require_assets(*model.required_assets(config))
    for name, expected in config.asset_sha256.items():
        path = config.require_assets(name)[name]
        if not path.is_file():
            raise ValueError(f"asset_sha256 requires a file, not a directory: {name}")
        if sha256(path) != expected:
            raise ValueError(f"Asset checksum mismatch: {name} ({path})")
    method_checks = getattr(model.implementation(), "validate", None)
    if method_checks is not None:
        method_checks(config, queries)


def predict(config: RunConfig, queries: list[dict]) -> list[dict]:
    """Return one prediction record per query, in query order."""
    queries = [normalize_query(q) for q in queries]
    validate(config, queries)
    config.output.mkdir(parents=True, exist_ok=True)
    rows = list(get_model(config.model).implementation().predict(config, queries))
    aligned = align_predictions(queries, rows)
    if len(rows) != len(queries) or any(r["status"] == "missing" for r in aligned):
        raise RuntimeError(f"{config.model} did not return exactly one prediction per query")
    lookup = {record_id(row): row for row in rows}
    return [lookup[q["id"]] for q in queries]


def run(config: RunConfig, *, limit: int | None = None) -> dict:
    """Predict for a query file and write predictions with their provenance."""
    queries = load_queries(config.queries)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        queries = queries[:limit]
    if (config.output / "predictions.jsonl").exists():
        raise FileExistsError(f"Output already contains predictions: {config.output}")
    config.output.mkdir(parents=True, exist_ok=True)
    payload = asdict(config) | {"queries": str(config.queries), "output": str(config.output)}
    report = {"model": config.model, "query_count": len(queries), "limit": limit, "config": payload}
    atomic_json(config.output / "inputs.json", input_record(config))
    atomic_json(config.output / "environment.json", environment_record())
    atomic_json(config.output / "run.json", report | {"status": "running"})
    try:
        predictions = predict(config, queries)
    except Exception as error:
        atomic_json(config.output / "run.json", report | {"status": "failed", "reason": str(error)})
        raise
    write_jsonl(config.output / "queries.jsonl", queries)
    write_jsonl(config.output / "predictions.jsonl", predictions)
    report |= {"status": "complete"}
    atomic_json(config.output / "run.json", report)
    return report


def doctor(config: RunConfig, queries: list[dict]) -> dict:
    """Check inputs and that this interpreter can import the model runtime."""
    validate(config, queries)
    model = get_model(config.model)
    dependencies = {m: importlib.util.find_spec(m) is not None for m in model.required_modules(config)}
    return {
        "model": config.model,
        "dataset": config.dataset,
        "queries": len(queries),
        "assets_present": model.required_assets(config),
        "dependencies": dependencies,
        "ready": all(dependencies.values()),
        "checkpoint_compatibility_checked": False,
    }
