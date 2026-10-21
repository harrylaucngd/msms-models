import csv
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("compare", ROOT / "paper/compare.py")
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)


def metrics(path, values):
    path.write_text(json.dumps({"metrics": {k: {"mean": v} for k, v in values.items()}}))
    return path


def test_compare_rounds_like_the_manuscript(tmp_path):
    good = metrics(tmp_path / "m.json", {
        "acc@1": 0.03292321713374345, "mces@1": 31.7708, "tanimoto@1": 0.30549,
        "acc@10": 0.0565049, "mces@10": 16.6699, "tanimoto@10": 0.39612,
    })
    assert compare.compare("table5", "mist_molforge", good) == []
    bad = metrics(tmp_path / "b.json", {
        "acc@1": 0.0334, "mces@1": 31.7708, "tanimoto@1": 0.30549,
        "acc@10": 0.0565049, "mces@10": 16.6699, "tanimoto@10": 0.39612,
    })
    assert compare.compare("table5", "mist_molforge", bad) == [("acc@1", "3.29", "3.34")]


def test_every_reproduced_row_points_to_a_runnable_folder():
    for table in ("table3", "table5"):
        for row in csv.DictReader((ROOT / f"paper/results/{table}.csv").open()):
            assert (row["origin"] == "cited") == (not row["folder"]), row["row"]
            folder = ROOT / "paper" / row["folder"]
            if row["folder"] and folder.exists():
                assert (folder / "run.sh").is_file() and (folder / "README.md").is_file(), folder
