from pathlib import Path

import pytest

from msms_models.methods.mist.legacy import load_features


def test_legacy_features_align_ids_without_target_columns(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    # Deliberately contains no structures or target fingerprints.
    rows = [{
        "identifier": ["second", "first"],
        "formula": "C2H6O",
        "precursor_mz": [47.049, 47.049],
        "adduct": ["[M+H]+", "[M+H]+"],
        "collision_energy": [20.0, 30.0],
        "peak_formula": [{"C2H6O": 1.0}, {"CH4": 0.5}],
    }]
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "features.parquet")
    queries = [{"id": "first", "formula": "C2H6O"}, {"id": "second", "formula": "C2H6O"}]
    features = load_features(Path(tmp_path), queries)
    assert [r["identifier"] for r in features] == ["first", "second"]
    assert [r["collision_energy"] for r in features] == [30.0, 20.0]
    from msms_models.datasets.prepare import prepare_queries
    from msms_models.io import load_queries

    output = tmp_path / "queries.jsonl"
    manifest = prepare_queries(tmp_path, output)
    assert manifest["queries"] == 2
    assert list(manifest["source_files_sha256"]) == ["features.parquet"]
    assert [q["id"] for q in load_queries(output)] == ["second", "first"]
    with pytest.raises(ValueError, match="formula differs"):
        load_features(tmp_path, [{"id": "first", "formula": "C3H8O"}])
    with pytest.raises(ValueError, match="Missing annotated"):
        load_features(tmp_path, [{"id": "missing", "formula": "C2H6O"}])
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "duplicate.parquet")
    with pytest.raises(ValueError, match="Duplicate annotated"):
        load_features(tmp_path, queries)
