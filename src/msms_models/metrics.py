"""Fixed-denominator molecule metrics and explicit benchmark protocols."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
from rdkit import Chem

from .chemistry import canonical_smiles, inchikey14, tanimoto2048, valid_smiles
from .io import align_predictions
from .provenance import environment_record


@dataclass(frozen=True)
class MetricProtocol:
    task: str = "de-novo"
    cutoffs: tuple[int, ...] = (1, 10)
    candidate_policy: str = "preserve"
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

    def validate(self):
        if self.task not in {"de-novo", "retrieval"}:
            raise ValueError("task must be de-novo or retrieval")
        if not self.cutoffs or any(k < 1 for k in self.cutoffs):
            raise ValueError("Metric cutoffs must be positive")
        if self.candidate_policy not in {"preserve", "unique-valid"}:
            raise ValueError("Unknown candidate policy")
        if self.task == "retrieval" and self.candidate_policy != "preserve":
            raise ValueError("Retrieval evaluation must preserve the supplied candidate universe")
        if self.mces_threshold < 0 or self.mces_time_limit <= 0:
            raise ValueError("Invalid MCES threshold or time limit")
        if self.mces_threads < 1:
            raise ValueError("MCES threads must be positive")
        if self.bootstrap_resamples < 0 or not 0 < self.confidence < 1:
            raise ValueError("Invalid bootstrap configuration")
        return self


def prepare_candidates(values, protocol: MetricProtocol):
    output = []
    seen = set()
    for value in values:
        valid = valid_smiles(
            value, connected=protocol.task == "de-novo", tautomer=protocol.tautomer
        )
        candidate = value if valid else None
        if protocol.candidate_policy == "unique-valid":
            key = inchikey14(candidate, protocol.tautomer)
            if key is None or key in seen:
                continue
            seen.add(key)
        output.append(candidate)
    return output


def make_mces(protocol: MetricProtocol) -> Callable[[str, str], float]:
    # Imports stay optional: ordinary Hit/Tanimoto evaluation needs no solver.
    import pulp
    from myopic_mces import MCES

    if protocol.mces_solver not in pulp.listSolvers(onlyAvailable=True):
        raise RuntimeError(f"MCES solver {protocol.mces_solver!r} is not installed")

    cache = None
    if protocol.mces_cache:
        path = Path(protocol.mces_cache)
        path.parent.mkdir(parents=True, exist_ok=True)
        cache = sqlite3.connect(path)
        cache.execute(
            "CREATE TABLE IF NOT EXISTS distances (protocol TEXT, target TEXT, predicted TEXT, "
            "value REAL, PRIMARY KEY(protocol, target, predicted))"
        )
    signature = json.dumps(
        {
            "schema": 1,
            "myopic_mces": "1.0.1",
            "threshold": protocol.mces_threshold,
            "stronger_bound": True,
            "seconds": protocol.mces_time_limit,
            "solver": protocol.mces_solver,
            "threads": protocol.mces_threads,
            "tautomer": protocol.tautomer,
        },
        sort_keys=True,
    )

    @lru_cache(maxsize=100_000)
    def distance(target, predicted):
        a = canonical_smiles(target, protocol.tautomer)
        b = canonical_smiles(predicted, protocol.tautomer)
        if a is None or b is None:
            raise ValueError("MCES requires valid molecular structures")
        if a == b:
            return 0.0
        if cache is not None:
            row = cache.execute(
                "SELECT value FROM distances WHERE protocol=? AND target=? AND predicted=?",
                (signature, a, b),
            ).fetchone()
            if row is not None:
                return float(row[0])
        try:
            result = MCES(
                a,
                b,
                threshold=protocol.mces_threshold,
                always_stronger_bound=True,
                solver=protocol.mces_solver,
                solver_options={
                    "msg": 0,
                    "timeLimit": protocol.mces_time_limit,
                    "threads": protocol.mces_threads,
                },
                catch_errors=False,
            )
        except OSError as error:
            raise RuntimeError(
                f"MCES solver {protocol.mces_solver!r} could not execute. "
                "If PuLP's bundled CBC has the wrong CPU architecture, install a native "
                "CBC executable on PATH and select --mces-solver COIN_CMD "
                "(MetricProtocol(mces_solver='COIN_CMD') in Python). See docs/install.md."
            ) from error
        value = float(result[1])
        if not np.isfinite(value) or value < 0:
            raise RuntimeError(f"MCES returned an invalid distance: {result}")
        if cache is not None:
            cache.execute(
                "INSERT OR REPLACE INTO distances VALUES (?, ?, ?, ?)", (signature, a, b, value)
            )
            cache.commit()
        return value

    return distance


def score_one(target: str, candidates, protocol: MetricProtocol, mces_fn=None) -> dict:
    protocol.validate()
    target_key = inchikey14(target, protocol.tautomer)
    if target_key is None:
        raise ValueError("The query manifest contains an invalid target SMILES")
    values = prepare_candidates(candidates, protocol)[: max(protocol.cutoffs)]
    keys = [inchikey14(v, protocol.tautomer) for v in values]
    similarity_values = values[:1] if protocol.task == "retrieval" else values
    similarity = [tanimoto2048(target, v, tautomer=protocol.tautomer) for v in similarity_values]
    result = {}
    prefix_name = "hit" if protocol.task == "retrieval" else "acc"
    for k in protocol.cutoffs:
        result[f"{prefix_name}@{k}"] = float(target_key in keys[:k])
        if protocol.task == "de-novo":
            result[f"tanimoto@{k}"] = max(similarity[:k], default=0.0)
    if protocol.task == "retrieval":
        result["tanimoto@1"] = similarity[0] if similarity else 0.0
    if protocol.mces:
        if mces_fn is None:
            mces_fn = make_mces(protocol)
        target_mol = Chem.MolFromSmiles(canonical_smiles(target, protocol.tautomer))
        empty_penalty = float(2 * target_mol.GetNumBonds())
        cutoffs = protocol.cutoffs if protocol.task == "de-novo" else (1,)
        distances = [
            mces_fn(target, v) if v is not None else empty_penalty for v in values[: max(cutoffs)]
        ]
        for k in cutoffs:
            result[f"mces@{k}"] = min(distances[:k], default=empty_penalty)
    return result


def bootstrap_mean(values, protocol: MetricProtocol) -> dict:
    from scipy.stats import bootstrap

    values = np.asarray(values, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Metrics must be finite for every query")
    output = {"mean": float(values.mean()), "n": len(values)}
    if not protocol.bootstrap_resamples:
        return output
    if len(values) == 1 or np.all(values == values[0]):
        output["ci"] = [float(values[0]), float(values[0])]
        output["ci_status"] = "constant_sample"
        return output
    result = bootstrap(
        (values,),
        np.mean,
        method="BCa",
        n_resamples=protocol.bootstrap_resamples,
        confidence_level=protocol.confidence,
        batch=128,
        random_state=protocol.seed,
    )
    bounds = [float(result.confidence_interval.low), float(result.confidence_interval.high)]
    if not np.isfinite(bounds).all():
        raise RuntimeError("BCa confidence interval is undefined for this sample")
    output["ci"] = bounds
    return output


def evaluate(queries: list[dict], predictions: list[dict], protocol=MetricProtocol()):
    protocol.validate()
    if not queries:
        raise ValueError("Evaluation requires a nonempty query manifest")
    ids = [row.get("id") for row in queries]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("Evaluation requires unique nonempty query ids")
    aligned = align_predictions(queries, predictions)
    mces_fn = make_mces(protocol) if protocol.mces else None
    details = [
        {
            "id": row["id"],
            "status": row["status"],
            **score_one(row["target_smiles"], row["candidates"], protocol, mces_fn),
        }
        for row in aligned
    ]
    columns = [k for k in details[0] if k not in {"id", "status"}]
    summary = {
        "schema_version": 1,
        "environment": environment_record(),
        "num_queries": len(queries),
        "missing_predictions": sum(row["status"] == "missing" for row in details),
        "protocol": {
            **asdict(protocol),
            "identity": "InChIKey14",
            "structure_fingerprint": "Morgan radius2/2048; achiral",
            "bootstrap_method": "BCa",
            "bootstrap_unit": "query/spectrum",
            "mces_always_stronger_bound": True,
        },
        "metrics": {k: bootstrap_mean([row[k] for row in details], protocol) for k in columns},
    }
    return summary, details
