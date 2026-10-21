"""Spectrum-blind stress-test baselines of paper Appendix A.4."""


from msms_models.config import RunConfig
from msms_models.data.io import write_jsonl
from msms_models.models.retrieval import chiral_atoms
from msms_models.models.retrieval.chemberta_classifier.model import text
from msms_models.models.retrieval.chemberta_classifier.train import training_pairs
from msms_models.models.retrieval.pubchem_order import inference as pubchem

QUERIES = [{"id": "q", "formula": "C4H10O", "target_smiles": "C[C@H](O)CC"}]
CANDIDATES = ["C[C@H](O)CC", "CCCCO", "CC(C)CO", "CCOCC"]


def config(tmp_path, model, **parameters):
    write_jsonl(tmp_path / "candidates.jsonl", [{"id": "q", "candidates": CANDIDATES}])
    assets = {"candidates": str(tmp_path / "candidates.jsonl")}
    if model == "pubchem-order":
        write_jsonl(tmp_path / "order.jsonl", [{
            "formula": "C4H10O", "cids": [1, 2], "inchikey14": ["LRHPLDYGYMQRHN", "ZXEKIIBDNHEJCQ"],
        }])
        assets["pubchem_order"] = str(tmp_path / "order.jsonl")
    return RunConfig(model, "msg", tmp_path / "q.jsonl", tmp_path / "out", assets=assets,
                     parameters=parameters, seed=3)


def test_chiral_atoms_counts_annotated_stereocentres():
    assert chiral_atoms.chiral_atoms("C[C@H](O)CC") == 1
    assert chiral_atoms.chiral_atoms("CC(O)CC") == 0
    assert chiral_atoms.fit_direction(QUERIES, {"q": CANDIDATES})["direction"] == "descending"


def test_chiral_ranking_breaks_ties_randomly_but_reproducibly(tmp_path):
    rows = chiral_atoms.predict(config(tmp_path, "chiral-atoms", direction="descending"), QUERIES)
    assert rows[0]["candidates"][0] == "C[C@H](O)CC"
    stable = chiral_atoms.predict(
        config(tmp_path, "chiral-atoms", direction="ascending", ties="stable"), QUERIES
    )
    assert stable[0]["candidates"] == ["CCCCO", "CC(C)CO", "CCOCC", "C[C@H](O)CC"]
    again = chiral_atoms.predict(config(tmp_path, "chiral-atoms", direction="ascending"), QUERIES)
    assert again == chiral_atoms.predict(
        config(tmp_path, "chiral-atoms", direction="ascending"), QUERIES
    )


def test_pubchem_order_ranks_listed_candidates_first(tmp_path):
    rows = pubchem.predict(config(tmp_path, "pubchem-order"), QUERIES)
    # n-butanol (LRHPLDYGYMQRHN) then isobutanol (ZXEKIIBDNHEJCQ) lead in PubChem order.
    assert rows[0]["candidates"][:2] == ["CCCCO", "CC(C)CO"]
    assert sorted(rows[0]["candidates"]) == sorted(CANDIDATES)


def test_classifier_pairs_one_molecule_one_decoy_and_canonicalize():
    queries = QUERIES * 2  # two spectra of one molecule give one pair
    pairs = training_pairs(queries, {"q": CANDIDATES}, canonical=True, seed=0)
    assert [label for _, label in pairs] == [1, 0]
    assert pairs[0][0] == text("C[C@H](O)CC", True) == "CC[C@H](C)O"
    assert pairs[1][0] in {text(c, True) for c in CANDIDATES[1:]}
