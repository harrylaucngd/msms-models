"""Prepare target-free candidate maps and explicit formula-known query inputs."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from .io import atomic_json, load_queries, write_jsonl
from .download import sha256

ADDUCT_DELTAS = {
    "[M+H]+": {"H": 1},
    "[M+Na]+": {"Na": 1},
    "[M+K]+": {"K": 1},
    "[M+NH4]+": {"N": 1, "H": 4},
}


def add_known_formula(query):
    """Declare the formula-known task; never called by an inference backend."""
    from rdkit import Chem
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula

    mol = Chem.MolFromSmiles(query.get("target_smiles", ""))
    if mol is None or not mol.GetNumAtoms():
        raise ValueError(f"Cannot derive formula for {query['id']}")
    formula = CalcMolFormula(mol)
    if query.get("formula") and query["formula"] != formula:
        raise ValueError(f"Provided formula differs from target structure for {query['id']}")
    query["formula"] = formula
    adduct = query.get("adduct")
    # The release can supply ion formulas with its own charge conventions.
    # Retain those annotations verbatim rather than replacing them heuristically.
    if adduct in ADDUCT_DELTAS and not query.get("precursor_formula"):
        # Formula tokens describe atom counts. Preserve the charged target formula
        # above, but do not interpret its trailing charge as an element count.
        composition = re.sub(r"[+-]\d*$", "", formula)
        tokens = re.findall(r"([A-Z][a-z]?)(\d*)", composition)
        if "".join(e + n for e, n in tokens) != composition:
            raise ValueError(f"{query['id']}: supply an explicit precursor_formula for {formula}")
        counts = {e: int(n or 1) for e, n in tokens}
        for element, delta in ADDUCT_DELTAS[adduct].items():
            counts[element] = counts.get(element, 0) + delta
        order = ["C", "H", *sorted(set(counts) - {"C", "H"})]
        precursor = "".join(
            e + (str(counts[e]) if counts[e] != 1 else "") for e in order if counts.get(e, 0)
        )
        query["precursor_formula"] = precursor
    return query


def prepare_queries(input_path, output, *, split=None, formula_from_target=False):
    queries = load_queries(input_path, split=split)
    if formula_from_target:
        for query in queries:
            add_known_formula(query)
    write_jsonl(output, queries)
    manifest = {
        "source": str(Path(input_path).resolve()),
        "output_sha256": sha256(output),
        "queries": len(queries),
        "split": split,
        "formula_from_target": formula_from_target,
        "precursor_formula_from_adduct": formula_from_target,
        "adduct_counts": dict(Counter(q.get("adduct", "missing") for q in queries)),
        "missing_precursor_formula": sum(not q.get("precursor_formula") for q in queries),
    }
    source = Path(input_path)
    if source.is_dir():
        manifest["source_files_sha256"] = {
            str(path.relative_to(source)): sha256(path)
            for path in sorted(source.rglob("*.parquet"))
        }
    else:
        manifest["source_sha256"] = sha256(source)
    atomic_json(Path(output).with_suffix(".manifest.json"), manifest)
    return manifest


def prepare_candidates(input_path, queries_path, output):
    from .candidates import load_candidates

    queries = load_queries(queries_path)
    library = load_candidates(input_path, queries)
    write_jsonl(output, ({"id": q["id"], "candidates": library[q["id"]]} for q in queries))
    manifest = {
        "source": str(Path(input_path).resolve()),
        "source_sha256": sha256(input_path),
        "queries_sha256": sha256(queries_path),
        "output_sha256": sha256(output),
        "queries": len(queries),
        "candidate_order": "preserved",
        "candidate_strings": "unchanged",
        "target_used_only_for_official_smiles_key_lookup": True,
    }
    atomic_json(Path(output).with_suffix(".manifest.json"), manifest)
    return manifest

