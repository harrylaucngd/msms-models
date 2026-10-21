"""MassSpecGym molecule metrics over a fixed query manifest."""

from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Callable

import numpy as np

from ..data.io import align_predictions
from .chemistry import canonical_smiles, inchikey14, tanimoto2048, valid_smiles

# MassSpecGym's distance for an invalid or missing prediction; per-spectrum minima
# are capped at this value.
MCES_INVALID = 100.0


@dataclass(frozen=True)
class MetricProtocol:
    task: str = "de-novo"  # or "retrieval"
    cutoffs: tuple[int, ...] = (1, 10)
    candidate_policy: str = "preserve"  # or "unique-valid" (de novo only)
    tautomer: bool = False
    mces: bool = False
    mces_threshold: int = 15
    mces_time_limit: float = 5.0
    mces_solver: str = "PULP_CBC_CMD"
    mces_threads: int = 1
    mces_cache: str | None = None
    bootstrap_resamples: int = 20_000
    confidence: float = 0.999
    seed: int = 0


def prepare_candidates(values, protocol: MetricProtocol):
    output, seen = [], set()
    for value in values:
        candidate = value if valid_smiles(value, tautomer=protocol.tautomer) else None
        if protocol.candidate_policy == "unique-valid":
            key = inchikey14(candidate, protocol.tautomer)
            if key is None or key in seen:
                continue
            seen.add(key)
        output.append(candidate)
    return output


def make_mces(protocol: MetricProtocol) -> Callable[[str, str], float]:
    """Myopic MCES distance with the stronger bound; optionally cached in SQLite."""
    from myopic_mces import MCES

    cache = None
    if protocol.mces_cache:
        cache = sqlite3.connect(protocol.mces_cache)
        cache.execute("CREATE TABLE IF NOT EXISTS distances (target TEXT, predicted TEXT, "
                      "value REAL, PRIMARY KEY(target, predicted))")

    @lru_cache(maxsize=100_000)
    def distance(target, predicted):
        a = canonical_smiles(target, protocol.tautomer)
        b = canonical_smiles(predicted, protocol.tautomer)
        if a == b:
            return 0.0
        if cache is not None:
            row = cache.execute("SELECT value FROM distances WHERE target=? AND predicted=?",
                                (a, b)).fetchone()
            if row is not None:
                return float(row[0])
        value = float(MCES(
            a, b, threshold=protocol.mces_threshold, always_stronger_bound=True,
            solver=protocol.mces_solver,
            solver_options={"msg": 0, "timeLimit": protocol.mces_time_limit,
                            "threads": protocol.mces_threads},
        )[1])
        if cache is not None:
            cache.execute("INSERT OR REPLACE INTO distances VALUES (?, ?, ?)", (a, b, value))
            cache.commit()
        return value

    return distance


def score_one(target: str, candidates, protocol: MetricProtocol, mces_fn=None) -> dict:
    target_key = inchikey14(target, protocol.tautomer)
    values = prepare_candidates(candidates, protocol)[: max(protocol.cutoffs)]
    keys = [inchikey14(v, protocol.tautomer) for v in values]
    retrieval = protocol.task == "retrieval"
    similarity = [tanimoto2048(target, v, tautomer=protocol.tautomer)
                  for v in (values[:1] if retrieval else values)]
    result = {}
    for k in protocol.cutoffs:
        result[f"{'hit' if retrieval else 'acc'}@{k}"] = float(target_key in keys[:k])
        if not retrieval:
            result[f"tanimoto@{k}"] = max(similarity[:k], default=0.0)
    if retrieval:
        result["tanimoto@1"] = similarity[0] if similarity else 0.0
    if protocol.mces:
        mces_fn = mces_fn or make_mces(protocol)
        cutoffs = (1,) if retrieval else protocol.cutoffs
        distances = [mces_fn(target, v) if v is not None else MCES_INVALID
                     for v in values[: max(cutoffs)]]
        for k in cutoffs:
            result[f"mces@{k}"] = min([MCES_INVALID, *distances[:k]])
    return result


def bootstrap_mean(values, protocol: MetricProtocol) -> dict:
    """Mean with a BCa confidence interval over spectrum-level resamples."""
    from scipy.stats import bootstrap

    values = np.asarray(values, dtype=np.float64)
    output = {"mean": float(values.mean()), "n": len(values)}
    if not protocol.bootstrap_resamples:
        return output
    if np.ptp(values) == 0:  # BCa is undefined for a constant sample
        output["ci"] = [output["mean"], output["mean"]]
        return output
    interval = bootstrap(
        (values,), np.mean, method="BCa", n_resamples=protocol.bootstrap_resamples,
        confidence_level=protocol.confidence, batch=128, random_state=protocol.seed,
    ).confidence_interval
    output["ci"] = [float(interval.low), float(interval.high)]
    return output


def evaluate(queries: list[dict], predictions: list[dict], protocol=MetricProtocol()):
    """Aggregate and per-spectrum metrics; every query counts, with or without a prediction."""
    mces_fn = make_mces(protocol) if protocol.mces else None
    details = [
        {"id": row["id"], "status": row["status"],
         **score_one(row["target_smiles"], row["candidates"], protocol, mces_fn)}
        for row in align_predictions(queries, predictions)
    ]
    columns = [k for k in details[0] if k not in {"id", "status"}]
    summary = {
        "num_queries": len(queries),
        "missing_predictions": sum(row["status"] == "missing" for row in details),
        "protocol": asdict(protocol),
        "metrics": {k: bootstrap_mean([row[k] for row in details], protocol) for k in columns},
    }
    return summary, details
