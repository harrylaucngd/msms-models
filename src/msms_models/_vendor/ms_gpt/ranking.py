# Adapted from VIKI623/MS-GPT, revision f2c844e61f07 and local paper protocol.
# Apache-2.0; see LICENSE.txt. Native candidate ordering is preserved.
from collections import Counter
import numpy as np
from rdkit import Chem
from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey
from rdkit.Chem import rdMolDescriptors, AllChem
from rdkit.DataStructs import ConvertToNumpyArray

def featurize_candidate(smi, fp_bits=4096):
    """SMILES → (ik14, formula, morgan_fp, canonical_smiles) or None."""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        Chem.RemoveStereochemistry(mol)
        mol2 = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        if mol2 is None:
            return None
        inchi = MolToInchi(mol2)
        if not inchi:
            return None
        ik = InchiToInchiKey(inchi).split("-")[0]
        formula = rdMolDescriptors.CalcMolFormula(mol2)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=fp_bits)
        arr = np.zeros(fp_bits, dtype=np.float32)
        ConvertToNumpyArray(fp, arr)
        return ik, formula, arr, Chem.MolToSmiles(mol2)
    except Exception:
        return None


def tanimoto_similarity(a, b):
    inter = (a * b).sum()
    union = ((a + b) > 0).sum()
    return float(inter / max(union, 1))


# Density-calibrated band for the band-marginalized tie-break similarity: a
# frequency tie is broken by max_t Tanimoto((posterior > t), candidate) over the
# density band -- the same posterior/fingerprint similarity used by the
# fidelity-binning analysis, replacing the single thresholded posterior reference.
BAND_POINTS = 20


def band_reference_queries(posterior, band_low, band_high, num=BAND_POINTS):
    thresholds = np.linspace(band_low, band_high, num, dtype=np.float32)
    return posterior[None, :] > thresholds[:, None]


def band_max_similarity(queries, query_popcounts, candidate_fp):
    candidate_bits = candidate_fp > 0
    intersection = (queries & candidate_bits).sum(1)
    union = query_popcounts + candidate_bits.sum() - intersection
    return float(np.max(intersection / np.maximum(union, 1)))


def rank_spectrum_candidates(args_tuple):
    """Parse + frequency_posterior_similarity rerank for one spectrum. Returns intermediate result."""
    record, formula_filter, posterior, band = args_tuple
    target_inchikey14 = record["target_inchikey14"]
    target_formula = record["target_formula"]

    band_queries = band_reference_queries(posterior, band[0], band[1])
    band_popcounts = band_queries.sum(1)

    total_valid = 0
    keys = Counter()
    ik_to_fps = {}
    ik_to_smi = {}

    for smi in record["candidate_smiles"]:
        r = featurize_candidate(smi)
        if r is None:
            continue
        ik, formula, fp_arr, canon = r
        total_valid += 1
        if formula_filter and target_formula and formula != target_formula:
            continue
        keys[ik] += 1
        ik_to_fps.setdefault(ik, []).append(fp_arr)
        if ik not in ik_to_smi:
            ik_to_smi[ik] = canon

    cands = []
    for k, freq in keys.items():
        sim = np.mean([
            band_max_similarity(band_queries, band_popcounts, fp)
            for fp in ik_to_fps[k]
        ])
        cands.append((k, freq, sim))
    cands.sort(key=lambda x: (-x[1], -x[2]))
    ranked = [c[0] for c in cands[:20]]

    hits = {}
    for k in (1, 5, 10, 20):
        hits[k] = 1 if target_inchikey14 in ranked[:k] else 0

    return {
        "cache_row": record["cache_row"],
        "target_inchikey14": target_inchikey14,
        "target_formula": target_formula,
        "total_generated": len(record["candidate_smiles"]),
        "total_valid": total_valid,
        "n_unique": len(keys),
        "hits_frequency_posterior_similarity": hits,
        "ranked_frequency_posterior_similarity": ranked,
        "ik_to_smi": ik_to_smi,
    }

