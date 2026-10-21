
from msms_models.data.io import load_queries, normalize_query, write_jsonl


def test_official_tsv_fields_have_stable_query_identity():
    row = normalize_query(
        {
            "identifier": "s1",
            "mzs": "30,10",
            "intensities": "2,1",
            "smiles": "CCO",
            "precursor_mz": "47.049",
            "fold": "test",
        }
    )
    assert row["id"] == "s1"
    assert row["spectrum"] == {"mz": [10, 30], "intensity": [1, 2]}
    assert row["precursor_mz"] == 47.049


def test_inference_does_not_require_the_unknown_target_structure():
    assert normalize_query({"id": "sample", "formula": "C2H6O"}) == {
        "id": "sample",
        "formula": "C2H6O",
    }


def test_split_selection(tmp_path):
    path = tmp_path / "queries.jsonl"
    write_jsonl(path, [{"id": "a", "split": "train"}, {"id": "b", "split": "test"}])
    assert [q["id"] for q in load_queries(path, split="test")] == ["b"]


def test_split_selection_does_not_parse_unselected_spectra(tmp_path):
    from msms_models.data.io import load_queries, write_jsonl

    source = tmp_path / "mixed.jsonl"
    write_jsonl(
        source,
        [
            {"id": "train", "split": "train", "spectrum": {"mz": [1.0], "intensity": []}},
            {"id": "test", "split": "test", "spectrum": {"mz": [2.0], "intensity": [1.0]}},
        ],
    )
    assert [row["id"] for row in load_queries(source, split="test")] == ["test"]
