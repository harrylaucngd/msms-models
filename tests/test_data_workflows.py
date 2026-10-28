import json

from msms_models.data.io import read_records, write_jsonl
from msms_models.data.prepare import add_known_formula, prepare_candidates, prepare_queries


def test_official_columns_and_ion_annotations_survive_preparation(tmp_path):
    source = tmp_path / "source.tsv"
    source.write_text(
        "identifier\tsmiles\tformula\tprecursor_formula\tadduct\tfold\tmzs\tintensities\n"
        "train\tCCO\tC2H6O\tC2H7O\t[M+H]+\ttrain\t31,29\t1,0.5\n"
        "test\tC[N+](C)(C)C\tC4H12N+\tC4H12N\t[M+H]+\ttest\t58\t1\n"
    )
    output = tmp_path / "queries.jsonl"
    report = prepare_queries(source, output, split="test", formula_from_target=True)
    rows = read_records(output)
    assert report["queries"] == 1
    assert rows[0]["id"] == "test"
    assert rows[0]["formula"] == "C4H12N+"
    assert rows[0]["precursor_formula"] == "C4H12N"


def test_known_formula_preparation_adds_the_precursor_formula():
    row = {"id": "q", "target_smiles": "CCO", "adduct": "[M+H]+"}
    assert add_known_formula(row)["precursor_formula"] == "C2H7O"


def test_candidate_conversion_preserves_strings_duplicates_and_ties(tmp_path):
    queries = tmp_path / "queries.jsonl"
    write_jsonl(queries, [{"id": "a", "target_smiles": "CCO"}])
    source = tmp_path / "candidates.json"
    source.write_text(json.dumps({"CCO": ["OCC", "COC", "OCC"]}))
    output = tmp_path / "converted.jsonl"
    prepare_candidates(source, queries, output)
    assert read_records(output) == [{"id": "a", "candidates": ["OCC", "COC", "OCC"]}]


def test_canonical_pool_falls_back_to_the_molecule_skeleton(tmp_path):
    from msms_models.data.candidates import load_candidates

    # "a" is keyed with a stereo assignment that matches neither the target nor its canonical form.
    source = tmp_path / "candidates.json"
    source.write_text(json.dumps({"C/C=C\\C": ["CC=CC"], "CCO": ["OCC"]}))
    queries = [{"id": "a", "target_smiles": "C/C=C/C"}, {"id": "b", "target_smiles": "OCC"}]
    assert load_candidates(source, queries) == {"a": ["CC=CC"], "b": ["OCC"]}


def test_preparation_keeps_only_the_listed_ids_in_their_order(tmp_path):
    import numpy as np

    write_jsonl(tmp_path / "q.jsonl", [{"id": i, "spectrum": {"mz": [1.0], "intensity": [1.0]}}
                                       for i in ("a", "b", "c")])
    np.save(tmp_path / "ids.npy", np.array(["c", "a"]))
    prepare_queries(tmp_path / "q.jsonl", tmp_path / "out.jsonl", ids=tmp_path / "ids.npy")
    assert [r["id"] for r in read_records(tmp_path / "out.jsonl")] == ["c", "a"]
