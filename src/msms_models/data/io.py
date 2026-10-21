"""Portable input and prediction records with a fixed, ordered query universe."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable


def record_id(record: dict) -> str:
    """Spectrum ID of a query, prediction or candidate record."""
    return str(record.get("id", record.get("identifier", record.get("spec", ""))))


def atomic_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_jsonl(path: str | Path, records: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
    temporary.replace(path)


def iter_records(path: str | Path) -> Iterable[dict]:
    path = Path(path)
    if path.is_dir() or path.suffix == ".parquet":
        from .annotated import iter_annotated_spectra

        for row in iter_annotated_spectra(path):
            # The released parquet uses NaN for unreported collision energies.
            energy = row.get("collision_energy")
            if isinstance(energy, float) and math.isnan(energy):
                row.pop("collision_energy")
            yield row
        return
    if path.suffix in {".tsv", ".csv"}:
        with path.open(newline="") as handle:
            yield from csv.DictReader(handle, delimiter="\t" if path.suffix == ".tsv" else ",")
        return
    if path.suffix == ".json":
        value = json.loads(path.read_text())
        if not isinstance(value, list):
            raise ValueError(f"{path}: expected a JSON list of records")
        yield from value
        return
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def read_records(path: str | Path) -> list[dict]:
    return list(iter_records(path))


def _numbers(value) -> list[float]:
    if value is None or (isinstance(value, str) and not value.strip()):
        return []
    if isinstance(value, str):
        value = json.loads(value) if value.lstrip().startswith("[") else value.split(",")
    values = [float(item) for item in value]
    if not all(math.isfinite(item) for item in values):
        raise ValueError("Spectrum arrays must contain finite values")
    return values


def normalize_query(record: dict, *, require_target: bool = False) -> dict:
    identifier = record_id(record)
    if not identifier.strip():
        raise ValueError("Every query requires a nonempty id")
    query = {"id": str(identifier)}
    target = record.get("target_smiles", record.get("smiles"))
    if require_target and not target:
        raise ValueError(f"Query {identifier}: target_smiles is required for evaluation")
    if target:
        query["target_smiles"] = str(target)
    spectrum = record.get("spectrum") or {}
    mz = _numbers(spectrum.get("mz", record.get("mzs")))
    intensity = _numbers(spectrum.get("intensity", record.get("intensities")))
    if len(mz) != len(intensity):
        raise ValueError(f"Query {identifier}: spectrum array lengths differ")
    if any(v <= 0 for v in mz) or any(v < 0 for v in intensity):
        raise ValueError(f"Query {identifier}: m/z must be positive and intensity nonnegative")
    if mz:
        order = sorted(range(len(mz)), key=mz.__getitem__)
        query["spectrum"] = {
            "mz": [mz[i] for i in order],
            "intensity": [intensity[i] for i in order],
        }
    aliases = {
        "formula": ("formula",),
        "precursor_formula": ("precursor_formula", "formula_corrected"),
        "precursor_mz": ("precursor_mz", "precursor_MZ_final"),
        "adduct": ("adduct", "ionization"),
        "instrument": ("instrument", "instrument_type"),
        "collision_energy": ("collision_energy",),
        "split": ("split", "fold"),
    }
    for field, names in aliases.items():
        value = next((record[n] for n in names if record.get(n) not in (None, "")), None)
        if value is not None:
            if field in {"precursor_mz", "collision_energy"}:
                value = float(value)
                if not math.isfinite(value):
                    raise ValueError(f"Query {identifier}: invalid {field}")
                if field == "precursor_mz" and value <= 0:
                    raise ValueError(f"Query {identifier}: precursor_mz must be positive")
            query[field] = value
    return query


def load_queries(path: str | Path, *, require_target: bool = False, split=None) -> list[dict]:
    records = [
        normalize_query(r, require_target=require_target)
        for r in iter_records(path)
        if split is None or r.get("split", r.get("fold")) == split
    ]
    if not records:
        raise ValueError("The query manifest is empty")
    ids = [r["id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate query ids are not allowed")
    return records


def candidate_smiles(record: dict) -> list[str | None]:
    values = record.get("candidates", record.get("ranked_smiles", record.get("predictions")))
    if values is None:
        columns = [k for k in record if k.startswith("pred_") or k.isdigit()]
        columns.sort(key=lambda k: int(k.rsplit("_", 1)[-1]))
        values = [record[k] for k in columns]
    if isinstance(values, str):
        values = json.loads(values)
    if not isinstance(values, list):
        raise ValueError("Predictions must contain an ordered candidate list")
    return [v.get("smiles") if isinstance(v, dict) else v for v in values]


def align_predictions(queries: list[dict], predictions: list[dict]) -> list[dict]:
    expected = {q["id"] for q in queries}
    lookup = {}
    for record in predictions:
        identifier = record_id(record)
        if identifier not in expected:
            raise ValueError(f"Prediction id {identifier!r} is outside the query manifest")
        if identifier in lookup:
            raise ValueError(f"Duplicate prediction id {identifier!r}")
        lookup[identifier] = record
    return [
        {
            "id": q["id"],
            "target_smiles": q.get("target_smiles"),
            "candidates": candidate_smiles(lookup[q["id"]]) if q["id"] in lookup else [],
            "status": lookup[q["id"]].get("status", "ok") if q["id"] in lookup else "missing",
        }
        for q in queries
    ]
