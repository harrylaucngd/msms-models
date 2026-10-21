"""Run each inference backend in its explicitly selected Python environment."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path

from .config import RunConfig
from .io import (
    align_predictions,
    atomic_json,
    load_queries,
    normalize_query,
    read_records,
    write_jsonl,
)
from .provenance import input_record
from .registry import get_model
from .validation import validate_run


def predict(config: RunConfig, queries):
    """Predict in the current process using already loaded query records.

    This is the same method call used by ``infer``'s isolated worker. It keeps
    method diagnostics in ``config.output`` and returns ordered prediction
    records. Use ``infer`` for file inputs, a selected interpreter and a complete
    on-disk run record.
    """
    queries = [normalize_query(q) for q in queries]
    identifiers = [q["id"] for q in queries]
    if not identifiers:
        raise ValueError("The query manifest is empty")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Duplicate query ids are not allowed")
    validate_run(config, queries)
    config.output.mkdir(parents=True, exist_ok=True)
    rows = list(get_model(config.model).predict(config, queries))
    aligned = align_predictions(queries, rows)
    if len(rows) != len(queries) or any(r["status"] == "missing" for r in aligned):
        raise RuntimeError("The method failed to account for every query")
    lookup = {str(row.get("id", row.get("identifier", row.get("spec", "")))): row for row in rows}
    return [lookup[q["id"]] for q in queries]


def infer(config: RunConfig, *, dry_run: bool = False):
    model = get_model(config.model)
    queries = load_queries(config.queries)
    validate_run(config, queries)
    if model.formula_required and any(not q.get("formula") for q in queries):
        raise ValueError(f"{model.name} requires a known molecular formula for every query")
    if config.output.exists() and (config.output / "predictions.jsonl").exists():
        raise FileExistsError(f"Output already contains predictions: {config.output}")
    payload = asdict(config)
    payload["queries"] = str(config.queries.resolve())
    payload["output"] = str(config.output.resolve())
    plan = {"model": model.to_dict(), "query_count": len(queries), "config": payload}
    if dry_run:
        return plan
    config.output.mkdir(parents=True, exist_ok=True)
    atomic_json(config.output / "inputs.json", input_record(config))
    request_path = config.output / "request.json"
    atomic_json(request_path, payload)
    env = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        env[key] = str(config.workers)
    env["TOKENIZERS_PARALLELISM"] = "false"
    if config.device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    command = [config.python, str(Path(__file__).with_name("_worker.py")), str(request_path)]
    atomic_json(config.output / "run.json", {**plan, "status": "running"})
    with (config.output / "inference.log").open("w") as log:
        process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    if process.returncode:
        atomic_json(
            config.output / "run.json",
            {
                **plan,
                "status": "failed",
                "returncode": process.returncode,
            },
        )
        raise RuntimeError(
            f"{config.model} inference failed; see {config.output / 'inference.log'}"
        )
    try:
        predictions = read_records(config.output / "predictions.jsonl")
        aligned = align_predictions(queries, predictions)
        if len(predictions) != len(queries) or any(r["status"] == "missing" for r in aligned):
            raise RuntimeError("The backend failed to account for every query")
    except Exception as error:
        atomic_json(config.output / "run.json", {**plan, "status": "failed", "reason": str(error)})
        raise
    write_jsonl(config.output / "queries.jsonl", queries)
    report = {**plan, "status": "complete", "prediction_count": len(predictions)}
    atomic_json(config.output / "run.json", report)
    return report


def load_request(path):
    payload = json.loads(Path(path).read_text())
    payload["queries"] = Path(payload["queries"])
    payload["output"] = Path(payload["output"])
    return RunConfig(**payload)
