"""Native ms-pred sparse entropy and candidate expansion (MIT).

Source: coleygroup/ms-pred, revision 708148c2a8eb0c32120644436fefd2fe90cd2214.
Only standalone scoring functions are retained. See LICENSE.txt.
"""
from typing import List, Dict, Sequence, Tuple
import numpy as np

def _sparse_indices_values(spec: np.ndarray, ignore_peak=None) -> Tuple[np.ndarray, np.ndarray]:
    if spec is None:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)

    if isinstance(spec, tuple) and len(spec) == 2:
        inds = np.asarray(spec[0], dtype=np.int64)
        vals = np.asarray(spec[1], dtype=np.float32)
        keep = vals > 0
        inds = inds[keep]
        vals = vals[keep]
        if ignore_peak is not None:
            keep = inds < int(ignore_peak)
            inds = inds[keep]
            vals = vals[keep]
        return inds, vals

    spec = np.asarray(spec)
    if spec.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
    if spec.ndim == 1:
        inds = np.flatnonzero(spec > 0).astype(np.int64, copy=False)
        vals = spec[inds].astype(np.float32, copy=False)
    else:
        inds = spec[:, 0].astype(np.int64, copy=False)
        vals = spec[:, 1].astype(np.float32, copy=False)
        keep = vals > 0
        inds = inds[keep]
        vals = vals[keep]

    if ignore_peak is not None:
        keep = inds < int(ignore_peak)
        inds = inds[keep]
        vals = vals[keep]
    return inds, vals

def _transform_sparse_intensities(vals: np.ndarray, sqrt_inten: bool = True) -> np.ndarray:
    """Return intensities in the space requested by the distance metric."""
    if sqrt_inten:
        return vals
    return np.square(vals, dtype=np.float32)

def _entropy_from_probs(vals: np.ndarray) -> float:
    vals = vals[vals > 0]
    return float(-np.sum(vals * np.log(vals + 1e-22)))

def _entropy_sparse(
    cand_preds_dict: List[Dict],
    true_spec: np.ndarray,
    colli_eng,
    ignore_peak=None,
    sqrt_inten=True,
) -> np.ndarray:
    true_inds, true_vals = _sparse_indices_values(true_spec, ignore_peak=ignore_peak)
    true_vals = _transform_sparse_intensities(true_vals, sqrt_inten=sqrt_inten)
    true_sum = float(true_vals.sum())
    if true_sum <= 0:
        return np.ones(len(cand_preds_dict), dtype=np.float32)

    q_vals = true_vals / true_sum
    q_lookup = {int(ind): float(val) for ind, val in zip(true_inds, q_vals)}
    h_true = _entropy_from_probs(q_vals)
    out = np.ones(len(cand_preds_dict), dtype=np.float32)

    for pred_idx, cand_dict in enumerate(cand_preds_dict):
        cand_pred = cand_dict.get(colli_eng)
        if cand_pred is None:
            continue
        cand_inds, cand_vals = _sparse_indices_values(cand_pred, ignore_peak=ignore_peak)
        cand_vals = _transform_sparse_intensities(cand_vals, sqrt_inten=sqrt_inten)
        pred_sum = float(cand_vals.sum())
        if pred_sum <= 0:
            continue

        p_vals = cand_vals / pred_sum
        h_pred = _entropy_from_probs(p_vals)
        h_mix = 0.0
        p_lookup = {}
        for ind, p_val in zip(cand_inds, p_vals):
            ind = int(ind)
            p_val = float(p_val)
            p_lookup[ind] = p_lookup.get(ind, 0.0) + p_val

        for ind, p_val in p_lookup.items():
            if ind not in q_lookup:
                mix_val = p_val / 2
                h_mix -= mix_val * np.log(mix_val + 1e-22)
        for ind, q_val in q_lookup.items():
            mix_val = (q_val + p_lookup.get(ind, 0.0)) / 2
            h_mix -= mix_val * np.log(mix_val + 1e-22)

        out[pred_idx] = (2 * h_mix - h_pred - h_true) / np.log(4)
    return out

def normalize_inchikey(ikey: str) -> str:
    return str(ikey).replace("ikey ", "").strip()

def stereo_group_key(ikey: str) -> str:
    ikey = normalize_inchikey(ikey)
    return ikey.split("-")[0]

def expand_scores_to_full_candidates(
    cand_ikeys: Sequence[str],
    dist: np.ndarray,
    full_candidate_ikeys: Sequence[str],
) -> np.ndarray:
    exact_map = {}
    stereo_map = {}
    for cand_ikey, cand_dist in zip(cand_ikeys, dist):
        exact_key = normalize_inchikey(cand_ikey)
        stereo_key = stereo_group_key(exact_key)
        exact_map[exact_key] = float(cand_dist)
        stereo_map[stereo_key] = float(cand_dist)

    expanded = np.empty(len(full_candidate_ikeys), dtype=np.float32)
    expanded.fill(np.nan)
    for idx, full_ikey in enumerate(full_candidate_ikeys):
        exact_key = normalize_inchikey(full_ikey)
        stereo_key = stereo_group_key(exact_key)
        if exact_key in exact_map:
            expanded[idx] = exact_map[exact_key]
        elif stereo_key in stereo_map:
            expanded[idx] = stereo_map[stereo_key]
    return expanded


def bin_spectrum(spec, num_bins: int = 15000, upper_limit: int = 1500, pool_fn: str = "add"):
        if hasattr(spec, "spec"):
            spec = spec.spec
        if spec is None or len(spec) == 0:
            return np.zeros(0, dtype=np.uint32), np.zeros(0, dtype=np.float32)

        spec = np.asarray(spec)
        mz = spec[:, 0]
        inten = spec[:, 1]
        scale = (num_bins - 1) / upper_limit
        bin_idx = np.floor(mz * scale).astype(np.int64) + 1
        valid = (bin_idx >= 0) & (bin_idx < num_bins)
        bin_idx = bin_idx[valid]
        inten = inten[valid].astype(np.float32, copy=False)
        if len(bin_idx) == 0:
            return np.zeros(0, dtype=np.uint32), np.zeros(0, dtype=np.float32)

        unique_inds, inverse = np.unique(bin_idx, return_inverse=True)
        if pool_fn == "add":
            vals = np.zeros(len(unique_inds), dtype=np.float32)
            np.add.at(vals, inverse, inten)
        elif pool_fn == "max":
            order = np.argsort(inverse, kind="stable")
            sorted_inverse = inverse[order]
            sorted_inten = inten[order]
            starts = np.r_[0, np.flatnonzero(np.diff(sorted_inverse)) + 1]
            vals = np.maximum.reduceat(sorted_inten, starts).astype(np.float32, copy=False)
        else:
            raise NotImplementedError()
        return unique_inds.astype(np.uint32, copy=False), vals.astype(np.float32, copy=False)
