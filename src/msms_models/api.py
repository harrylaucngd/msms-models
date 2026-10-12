"""Run each inference backend in its explicitly selected Python environment."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path

from .config import RunConfig
from .io import align_predictions, atomic_json, load_queries, read_records, write_jsonl
from .registry import get_model
from .validation import validate_run


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
