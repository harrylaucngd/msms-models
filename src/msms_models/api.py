"""Run a configured model on a query file."""

from __future__ import annotations

from .config import RunConfig
from .data.io import load_queries, normalize_query, record_id, write_jsonl
from .models import get_model


def predict(config: RunConfig, queries: list[dict]) -> list[dict]:
    """Return one prediction record per query, in query order."""
    queries = [normalize_query(q) for q in queries]
    config.output.mkdir(parents=True, exist_ok=True)
    rows = get_model(config.model).implementation().predict(config, queries)
    lookup = {record_id(row): row for row in rows}
    return [lookup[q["id"]] for q in queries]


def run(config: RunConfig, *, limit: int | None = None) -> dict:
    """Predict for the configured query file and write the queries and predictions."""
    queries = load_queries(config.queries)[:limit]
    predictions = predict(config, queries)
    write_jsonl(config.output / "queries.jsonl", queries)
    write_jsonl(config.output / "predictions.jsonl", predictions)
    return {"model": config.model, "queries": len(queries), "output": str(config.output)}
