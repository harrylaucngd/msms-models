import hashlib
import json

import pytest

from msms_models.cli import main
from msms_models.datasets.download import fetch
from msms_models.datasets.prepare import add_known_formula, prepare_candidates, prepare_queries
from msms_models.io import read_records, write_jsonl


def test_download_is_verified_atomic_and_reusable(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"benchmark\n")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    target = tmp_path / "downloads" / "dataset"
    fetch(source.as_uri(), target, expected_sha256=digest, expected_size=10)
    source.unlink()
    assert fetch(source.as_uri(), target, expected_sha256=digest) == target
    target.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="existing"):
        fetch(source.as_uri(), target, expected_sha256=digest)


def test_corrupt_download_does_not_leave_a_dataset(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"incomplete")
    target = tmp_path / "downloads" / "dataset"
    with pytest.raises(ValueError, match="checksum"):
        fetch(source.as_uri(), target, expected_sha256="0" * 64)
    assert not target.exists()
    assert list(target.parent.iterdir()) == []


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


def test_known_formula_preparation_is_explicit_and_preserves_release_annotations():
    row = {"id": "q", "target_smiles": "CCO", "adduct": "[M+H]+"}
    assert add_known_formula(row)["precursor_formula"] == "C2H7O"
    with pytest.raises(ValueError, match="differs"):
        add_known_formula({"id": "q", "target_smiles": "CCO", "formula": "CH4"})


def test_candidate_conversion_preserves_strings_duplicates_and_ties(tmp_path):
    queries = tmp_path / "queries.jsonl"
    write_jsonl(queries, [{"id": "a", "target_smiles": "CCO"}])
    source = tmp_path / "candidates.json"
    source.write_text(json.dumps({"CCO": ["OCC", "COC", "OCC"]}))
    output = tmp_path / "converted.jsonl"
    prepare_candidates(source, queries, output)
    assert read_records(output) == [{"id": "a", "candidates": ["OCC", "COC", "OCC"]}]


def test_missing_input_explains_how_to_download(tmp_path, capsys):
    with pytest.raises(SystemExit):
        main(["prepare", "--input", str(tmp_path / "missing.tsv"), "--output", "unused"])
    assert "msms-models download msg" in capsys.readouterr().err
