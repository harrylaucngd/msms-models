"""Prepare query records and ID-keyed candidate pools from the official files."""

from __future__ import annotations

import re

from .io import load_queries, write_jsonl

ADDUCT_DELTAS = {
    "[M+H]+": {"H": 1},
    "[M+Na]+": {"Na": 1},
    "[M+K]+": {"K": 1},
    "[M+NH4]+": {"N": 1, "H": 4},
}


def add_known_formula(query):
    """The bonus (formula-known) challenge: take the formula from the target structure."""
    from rdkit import Chem
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula

    formula = CalcMolFormula(Chem.MolFromSmiles(query["target_smiles"]))
    query["formula"] = formula
    adduct = query.get("adduct")
    if adduct in ADDUCT_DELTAS and not query.get("precursor_formula"):
        # Count atoms of the neutral formula (ignoring a trailing charge) and add the adduct.
        tokens = re.findall(r"([A-Z][a-z]?)(\d*)", re.sub(r"[+-]\d*$", "", formula))
        counts = {e: int(n or 1) for e, n in tokens}
        for element, delta in ADDUCT_DELTAS[adduct].items():
            counts[element] = counts.get(element, 0) + delta
        order = ["C", "H", *sorted(set(counts) - {"C", "H"})]
        query["precursor_formula"] = "".join(
            e + (str(counts[e]) if counts[e] != 1 else "") for e in order if counts.get(e, 0)
        )
    return query


def prepare_queries(input_path, output, *, split=None, formula_from_target=False):
    queries = load_queries(input_path, split=split)
    if formula_from_target:
        queries = [add_known_formula(q) for q in queries]
    write_jsonl(output, queries)
    return {"queries": len(queries), "output": str(output)}


def prepare_candidates(input_path, queries_path, output):
    from .candidates import load_candidates

    queries = load_queries(queries_path)
    library = load_candidates(input_path, queries)
    write_jsonl(output, ({"id": q["id"], "candidates": library[q["id"]]} for q in queries))
    return {"queries": len(queries), "output": str(output)}
