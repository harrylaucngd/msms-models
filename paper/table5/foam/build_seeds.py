"""Build FOAM's seed lists: DiffMS suggestions plus formula-matched PubChem structures.

For each test spectrum the seeds are the structures DiffMS proposes for it and the PubChem
structures with the query's formula (mixtures excluded), leaving out PubChem structures
whose 2D InChIKey belongs to a MassSpecGym test molecule. Every seed is tautomerized by an
InChI round trip without stereochemistry, as FOAM does, and duplicates are dropped.

    python paper/table5/foam/build_seeds.py --queries data/msg/queries.jsonl \\
        --diffms results/diffms/predictions.jsonl --pubchem data/pubchem/CID-SMILES.gz \\
        --output data/paper/table5/foam/seeds.jsonl

``--diffms`` holds DiffMS predictions as ``{"id", "candidates": [SMILES, ...]}`` records;
``--pubchem`` is PubChem's ``CID-SMILES`` listing (tab-separated CID and SMILES).
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from multiprocessing import Pool

from rdkit import Chem, RDLogger
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

from msms_models.data.io import load_queries, read_records, write_jsonl
from msms_models.simulators import scatter

RDLogger.DisableLog("rdApp.*")
# ms-pred imports torch_scatter for its reductions; ..simulators.scatter provides them.
sys.modules.setdefault("torch_scatter", scatter)


def tautomerize(smiles):
    """FOAM's ``tautomerize_smi``: an InChI round trip, written without stereochemistry."""
    from ms_pred.common.chem_utils import canonical_mol_from_inchi

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = canonical_mol_from_inchi(Chem.MolToInchi(mol))
    if mol is None:
        return None
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol)


def inchikey2d(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToInchiKey(mol)[:14] if mol is not None else None


def formula_entry(line):
    """(uncharged formula, SMILES) of one PubChem line, or None."""
    smiles = line.rstrip("\n").split("\t")[-1]
    if "." in smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return re.match(r"^([^+\-]*)", CalcMolFormula(mol)).group(1), smiles


def pubchem_by_formula(path, formulas, workers):
    opener = gzip.open if str(path).endswith(".gz") else open
    found = {f: [] for f in formulas}
    with opener(path, "rt") as handle, Pool(workers) as pool:
        for entry in pool.imap(formula_entry, handle, chunksize=20_000):
            if entry is not None and entry[0] in found:
                found[entry[0]].append(entry[1])
    return found


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--queries", required=True)
    parser.add_argument("--diffms", required=True)
    parser.add_argument("--pubchem", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)

    queries = load_queries(args.queries)
    test_keys = {inchikey2d(q["target_smiles"]) for q in queries}
    diffms = {row["id"]: row["candidates"] for row in read_records(args.diffms)}
    pubchem = pubchem_by_formula(args.pubchem, {q["formula"] for q in queries}, args.workers)

    with Pool(args.workers) as pool:
        library = {f: [s for s, key in zip(smiles, pool.map(inchikey2d, smiles, chunksize=256))
                       if key not in test_keys]
                   for f, smiles in pubchem.items()}
        rows = []
        for query in queries:
            raw = [s for s in diffms.get(query["id"], []) if s] + library[query["formula"]]
            seeds = sorted({s for s in pool.map(tautomerize, raw, chunksize=256) if s})
            rows.append({"id": query["id"], "candidates": seeds})
    write_jsonl(args.output, rows)
    print(json.dumps({"queries": len(rows), "output": args.output}))


if __name__ == "__main__":
    main()
