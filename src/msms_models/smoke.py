"""Exercise a configured method through the same worker used for a full run."""

from dataclasses import replace
from pathlib import Path

from .api import infer
from .config import RunConfig
from .datasets.download import sha256
from .io import atomic_json, load_queries, read_records, write_jsonl
from .metrics import MetricProtocol, evaluate
from .registry import get_model


def smoke(config: RunConfig, output, *, limit=2, device=None):
    if limit < 1:
        raise ValueError("limit must be positive")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Choose a fresh smoke output directory: {output}")
    queries = load_queries(config.queries)[:limit]
    output.mkdir(parents=True)
    query_path = output / "queries.jsonl"
    write_jsonl(query_path, queries)
    run_config = replace(
        config, queries=query_path, output=output / "inference", device=device or config.device
    )
    try:
        result = infer(run_config)
        predictions = read_records(run_config.output / "predictions.jsonl")
        if not any(p.get("candidates") for p in predictions):
            raise RuntimeError(
                "All smoke queries returned empty predictions; inspect their statuses"
            )
        task = get_model(config.model).task
        metric_result = None
        if all(q.get("target_smiles") for q in queries):
            metric_result, details = evaluate(
                queries,
                predictions,
                MetricProtocol(
                    task=task,
                    cutoffs=(1, 5, 20) if task == "retrieval" else (1, 10),
                    bootstrap_resamples=0,
                ),
            )
            atomic_json(output / "metrics.json", metric_result)
            write_jsonl(output / "metrics.details.jsonl", details)
        report = {
            "status": "passed",
            "model": config.model,
            "dataset": config.dataset,
            "scope": "Inference and metric plumbing on a subset; no full-benchmark accuracy claim.",
            "query_count": len(queries),
            "source_queries_sha256": sha256(config.queries),
            "device": run_config.device,
            "parameters": config.parameters,
            "run": result,
            "metrics_computed": metric_result is not None,
        }
    except Exception as error:
        atomic_json(
            output / "smoke.json", {"status": "failed", "model": config.model, "error": str(error)}
        )
        raise
    atomic_json(output / "smoke.json", report)
    return report
