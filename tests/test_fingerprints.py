import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from msms_models.posteriors import align_posterior
from msms_models.retrieval.fingerprints import rank_fingerprints


def test_posterior_rows_are_aligned_by_id_and_missing_rows_fail():
    values = np.stack([np.zeros(4096), np.ones(4096)])
    out = align_posterior(values, ["b", "a"], [{"id": "a"}, {"id": "b"}])
    assert out[0].sum() == 4096 and out[1].sum() == 0
    with pytest.raises(ValueError, match="missing"):
        align_posterior(values, ["b", "a"], [{"id": "missing"}])


def test_ties_preserve_source_order_and_fingerprint_ranking_matches_structure():
    fp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096).GetFingerprintAsNumPy(
        Chem.MolFromSmiles("CCO")
    )
    rows = rank_fingerprints([{"id": "s"}], fp[None], {"s": ["CCN", "CCO", "OCC"]})
    assert rows[0]["candidates"][:2] == ["CCO", "OCC"]
    assert rows[0]["scores"][:2] == [1.0, 1.0]


def test_invalid_candidate_is_not_silently_removed():
    with pytest.raises(ValueError, match="Invalid candidate"):
        rank_fingerprints([{"id": "s"}], np.zeros((1, 4096)), {"s": ["not a molecule"]})
