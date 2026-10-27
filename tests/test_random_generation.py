import pickle
import subprocess
import sys
from collections import Counter
from pathlib import Path

from rdkit import Chem

from msms_models.config import RunConfig
from msms_models.models.de_novo.random_generation.fit import fit
from msms_models.models.de_novo.random_generation.inference import predict

TRAIN = [{"id": f"t{i}", "target_smiles": s}
         for i, s in enumerate(["CCO", "OCC(=O)O", "CC(N)C(=O)O", "c1ccccc1O"])]


def config(tmp_path, seed=0, **parameters):
    with (tmp_path / "stats.pkl").open("wb") as handle:
        pickle.dump(fit(TRAIN), handle)
    return RunConfig("random-generation", "msg", tmp_path / "q.jsonl", tmp_path / "out",
                     assets={"element_stats": str(tmp_path / "stats.pkl")},
                     parameters={"estimate_chem_element_stats": True, **parameters}, seed=seed)


def test_fit_records_element_statistics_and_formula_masses():
    stats = fit(TRAIN)
    assert {"C", "O", "H", "N"} <= set(stats["element_2_bond_stats"])
    assert len(stats["mol_weight_trn_values"]) == len(TRAIN)


def test_molecules_have_the_query_heavy_atoms(tmp_path):
    # As upstream, molecules are written unsanitized and can fall apart when no spanning tree
    # is found; parsing them adds implicit hydrogens, so heavy atoms are compared.
    queries = [{"id": "a", "formula": "C2H6O"}, {"id": "b", "formula": "C3H7NO2"}]
    expected = [{"C": 2, "O": 1}, {"C": 3, "N": 1, "O": 2}]
    rows = predict(config(tmp_path), queries)
    assert [len(r["candidates"]) for r in rows] == [10, 10]
    for heavy, row in zip(expected, rows):
        for smiles in row["candidates"]:
            atoms = Counter(a.GetSymbol() for a in Chem.MolFromSmiles(smiles).GetAtoms())
            assert {k: v for k, v in atoms.items() if k != "H"} == heavy


def test_generation_is_seeded_and_cached_per_formula(tmp_path):
    queries = [{"id": "a", "formula": "C3H7NO2"}, {"id": "b", "formula": "C3H7NO2"}]
    first = predict(config(tmp_path, enforce_connectivity=False), queries)
    assert first == predict(config(tmp_path, enforce_connectivity=False), queries)
    assert first[0]["candidates"] == first[1]["candidates"]
    other = predict(config(tmp_path, seed=1, enforce_connectivity=False), queries)
    assert other[0]["candidates"] != first[0]["candidates"]


def test_inference_import_does_not_load_torch():
    root = str(Path(__file__).resolve().parents[1] / "src")
    subprocess.run([sys.executable, "-c",
                    "import sys; sys.path.insert(0, sys.argv[1]); "
                    "from msms_models.models.de_novo.random_generation import inference, fit; "
                    "assert 'torch' not in sys.modules", root], check=True)
