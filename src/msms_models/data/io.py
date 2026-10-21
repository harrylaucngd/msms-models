"""Query, candidate and prediction records."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable

FIELDS = {
    "formula": ("formula",),
    "precursor_formula": ("precursor_formula", "formula_corrected"),
    "precursor_mz": ("precursor_mz", "precursor_MZ_final"),
    "adduct": ("adduct", "ionization"),
    "instrument": ("instrument", "instrument_type"),
    "collision_energy": ("collision_energy",),
    "split": ("split", "fold"),
}


def record_id(record: dict) -> str:
    """Spectrum ID of a query, prediction or candidate record."""
    return str(record.get("id", record.get("identifier", record.get("spec", ""))))


def write_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def write_jsonl(path: str | Path, records: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def iter_records(path: str | Path) -> Iterable[dict]:
    path = Path(path)
    if path.is_dir() or path.suffix == ".parquet":
        from .annotated import iter_annotated_spectra

        yield from iter_annotated_spectra(path)
    elif path.suffix in {".tsv", ".csv"}:
        with path.open(newline="") as handle:
            yield from csv.DictReader(handle, delimiter="\t" if path.suffix == ".tsv" else ",")
    elif path.suffix == ".json":
        yield from json.loads(path.read_text())
    else:
        with path.open() as handle:
            yield from (json.loads(line) for line in handle if line.strip())


def read_records(path: str | Path) -> list[dict]:
    return list(iter_records(path))


def _numbers(value) -> list[float]:
    if value is None or (isinstance(value, str) and not value.strip()):
        return []
    if isinstance(value, str):
        value = json.loads(value) if value.lstrip().startswith("[") else value.split(",")
    return [float(item) for item in value]


def normalize_query(record: dict) -> dict:
    """A query record with the standard field names and peaks sorted by m/z."""
    query = {"id": record_id(record)}
    target = record.get("target_smiles", record.get("smiles"))
    if target:
        query["target_smiles"] = str(target)
    spectrum = record.get("spectrum") or {}
    mz = _numbers(spectrum.get("mz", record.get("mzs")))
    intensity = _numbers(spectrum.get("intensity", record.get("intensities")))
    if mz:
        order = sorted(range(len(mz)), key=mz.__getitem__)
        query["spectrum"] = {"mz": [mz[i] for i in order], "intensity": [intensity[i] for i in order]}
    for field, names in FIELDS.items():
        value = next((record[n] for n in names if record.get(n) not in (None, "")), None)
        if field in {"precursor_mz", "collision_energy"} and value is not None:
            value = float(value)
            # Exported tables mark unreported collision energies as NaN.
            value = None if math.isnan(value) else value
        if value is not None:
            query[field] = value
    return query


def load_queries(path: str | Path, *, split=None) -> list[dict]:
    return [
        normalize_query(r) for r in iter_records(path)
        if split is None or r.get("split", r.get("fold")) == split
    ]


def candidate_smiles(record: dict) -> list[str | None]:
    values = record.get("candidates", record.get("ranked_smiles", record.get("predictions")))
    if values is None:  # wide tables with pred_smiles_1 ... pred_smiles_k columns
        columns = sorted((k for k in record if k.startswith("pred_")),
                         key=lambda k: int(k.rsplit("_", 1)[-1]))
        values = [record[k] for k in columns]
    if isinstance(values, str):
        values = json.loads(values)
    return [v.get("smiles") if isinstance(v, dict) else v for v in values]


def align_predictions(queries: list[dict], predictions: list[dict]) -> list[dict]:
    """Predictions in query order; a query without a prediction gets an empty list."""
    lookup = {record_id(record): record for record in predictions}
    return [
        {
            "id": q["id"],
            "target_smiles": q.get("target_smiles"),
            "candidates": candidate_smiles(lookup[q["id"]]) if q["id"] in lookup else [],
            "status": lookup[q["id"]].get("status", "ok") if q["id"] in lookup else "missing",
        }
        for q in queries
    ]
