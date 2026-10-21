"""Freeze PubChem's default compound order for every query formula.

    python -m msms_models.models.retrieval.pubchem_order.build \\
        --queries data/msg/queries.jsonl --output data/paper/table3/pubchem_order.jsonl

Each output row is ``{"formula", "cids", "inchikey14"}`` in the order returned by
PUG-REST ``compound/fastformula``. PubChem changes over time, so the frozen file,
not this script, defines a reported result.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from ....data.io import load_queries, read_records

PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/fastformula/{}/property/InChIKey/JSON"


def listing(formula: str, *, retries: int = 5) -> dict:
    url = PUG.format(urllib.parse.quote(formula))
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                properties = json.load(response)["PropertyTable"]["Properties"]
            break
        except urllib.error.HTTPError as error:
            if error.code == 404:  # PubChem has no compound with this formula
                properties = []
                break
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)
    return {
        "formula": formula,
        "cids": [p["CID"] for p in properties],
        "inchikey14": [p["InChIKey"][:14] for p in properties],
    }


def build(queries, output: Path, *, pause: float = 0.25):
    """Append missing formulas to ``output`` so an interrupted build resumes."""
    done = {r["formula"] for r in read_records(output)} if output.exists() else set()
    formulas = sorted({q["formula"] for q in queries} - done)
    with output.open("a") as handle:
        for formula in formulas:
            handle.write(json.dumps(listing(formula)) + "\n")
            handle.flush()
            time.sleep(pause)  # PubChem allows at most five requests per second
    return {"formulas": len(done) + len(formulas), "fetched": len(formulas)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(build(load_queries(args.queries), args.output)))


if __name__ == "__main__":
    main()
