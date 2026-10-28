"""Freeze SpecBridge's evaluation candidate order into an ID-keyed candidate file.

    python -m msms_models.models.retrieval.dreams_chemberta.candidates \
        --library MassSpecGym_retrieval_candidates_formula.json \
        --queries data/msg/queries.jsonl --output data/paper/table3/dreams_chemberta/candidates.jsonl

SpecBridge's evaluation seeds Python's `random` with 1234 and then, for every key
of the target-keyed pool *in file order* (all folds), appends the key if absent,
shuffles the list in place and removes duplicates. The official pools list the
true molecule first, so this order decides exactly tied scores. Like
`msms-models prepare-candidates`, this step reads the target only to select the
pool; inference reads the frozen, ID-keyed result.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path

from ....data.io import load_queries, write_jsonl


def shuffled_pools(library, seed=1234):
    """Per key: drop empties, append the key, shuffle, deduplicate."""
    rng = random.Random(seed)
    pools = {}
    for key, values in library.items():
        # Strings are used as given.
        vals = [v for v in values if v]
        if key not in vals:
            vals.append(key)
        rng.shuffle(vals)
        pools[key] = list(dict.fromkeys(vals))
    return pools


def read_library(path):
    path = Path(path)
    if path.suffix == ".pkl":
        # SpecBridge's released pools are pickled dicts; only open files you trust.
        with path.open("rb") as handle:
            return pickle.load(handle)
    return json.loads(path.read_text())


def freeze_candidates(library_path, queries_path, output, *, seed=1234, canonical_keys=False):
    library = read_library(library_path)
    pools = shuffled_pools(library, seed)
    queries = load_queries(queries_path)
    canon = None
    if canonical_keys:
        from rdkit import Chem

        def canon(smiles):
            return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))

    rows = []
    for query in queries:
        key = canon(query["target_smiles"]) if canon else query["target_smiles"]
        rows.append({"id": query["id"], "candidates": pools[key]})
    write_jsonl(output, rows)
    return {"queries": len(rows), "output": str(output)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--library", type=Path, required=True, help="Target-keyed pool (.json/.pkl)")
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--canonical-keys", action="store_true",
        help="Look pools up by the RDKit-canonical target (the `_ca` pool's keys)",
    )
    args = parser.parse_args(argv)
    summary = freeze_candidates(
        args.library, args.queries, args.output, seed=args.seed,
        canonical_keys=args.canonical_keys,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
