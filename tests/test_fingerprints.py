import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from msms_models.encoders.mist import align_posterior
from msms_models.models.retrieval.fingerprint import rank_fingerprints


def test_posterior_rows_are_aligned_by_id():
    values = np.stack([np.zeros(4096), np.ones(4096)])
    out = align_posterior(values, ["b", "a"], [{"id": "a"}, {"id": "b"}])
    assert out[0].sum() == 4096 and out[1].sum() == 0


def test_ties_preserve_source_order_and_fingerprint_ranking_matches_structure():
    fp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096).GetFingerprintAsNumPy(
        Chem.MolFromSmiles("CCO")
    )
    rows = rank_fingerprints([{"id": "s"}], fp[None], {"s": ["CCN", "CCO", "OCC"]})
    assert rows[0]["candidates"][:2] == ["CCO", "OCC"]
    assert rows[0]["scores"][:2] == [1.0, 1.0]


def test_mist_feature_rows_match_ids_with_acquisition_suffixes():
    from msms_models.encoders.mist.scaled import select_feature_rows

    assert select_feature_rows(
        ["b_collision 20", "a"], [{"id": "a"}, {"id": "b"}]
    ) == [1, 0]
