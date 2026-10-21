"""Compare a metrics.json written by `msms-models evaluate` with a printed table row.

    python paper/compare.py table3 mist_4m_aug results/paper/table3/mist_4m_aug/metrics.json

Percentages are compared after rounding to the printed precision. Confidence
interval bounds depend on the bootstrap seed and resampling order recorded in
each row's README; they are checked when the metrics file contains them.
"""

import argparse
import csv
import json
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
PERCENT = ("hit@", "acc@")


def printed(value: float, reference: str, percent: bool) -> str:
    """Round like the manuscript: same decimals as the printed reference."""
    places = Decimal(reference).as_tuple().exponent
    scaled = Decimal(repr(value * 100 if percent else value))
    return str(scaled.quantize(Decimal(1).scaleb(places), rounding=ROUND_HALF_UP))


def compare(table: str, row: str, metrics_path: Path) -> list[tuple[str, str, str]]:
    rows = {r["row"]: r for r in csv.DictReader((RESULTS / f"{table}.csv").open())}
    if row not in rows:
        raise SystemExit(f"Unknown {table} row {row!r}; choose from {', '.join(rows)}")
    expected = rows[row]
    metrics = json.loads(metrics_path.read_text())["metrics"]
    mismatches = []
    for column, reference in expected.items():
        if column in {"row", "method", "origin", "folder"} or not reference:
            continue
        name, _, bound = column.partition("_")
        if name not in metrics:
            mismatches.append((column, reference, "missing"))
            continue
        if bound:
            if "ci" not in metrics[name]:
                continue
            value = metrics[name]["ci"][0 if bound == "low" else 1]
        else:
            value = metrics[name]["mean"]
        got = printed(value, reference, name.startswith(PERCENT))
        if got != reference:
            mismatches.append((column, reference, got))
    return mismatches


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("table", choices=["table3", "table5"])
    parser.add_argument("row")
    parser.add_argument("metrics", type=Path)
    args = parser.parse_args(argv)
    mismatches = compare(args.table, args.row, args.metrics)
    for column, reference, got in mismatches:
        print(f"{args.table}/{args.row} {column}: printed {reference}, reproduced {got}")
    if mismatches:
        sys.exit(1)
    print(f"{args.table}/{args.row}: every printed value reproduced")


if __name__ == "__main__":
    main()
