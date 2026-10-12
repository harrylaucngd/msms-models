"""ms-pred 00948ec native dense distance (MIT); used by FOAM."""
import copy
from typing import List,Dict
import numpy as np
from numpy.linalg import norm

def dist_bin(cand_preds_dict: List[Dict], true_spec_dict: dict, sparse=True, ignore_peak=None, func='cos', selected_evs=None, agg=True) -> np.ndarray:
    """cos_dist for binned spectrum

    Args:
        cand_preds_dict: List of candidates
        true_spec_dict:
        ignore_peak: ignore peaks whose indices are larger than this value

    Returns:
        np.ndarray:
    """
    dist = []
    true_npeaks = []
    ## sampled_evs = np.random.choice(evs, 3, p = ())
    if selected_evs:
        true_spec_dict = {k: v for k, v in true_spec_dict.items() if str(k) in selected_evs}

    # standardize keys to str
    true_spec_dict = {f'{float(k):.0f}': v for k, v in true_spec_dict.items()}
    cand_preds_dict = [{f'{float(k):.0f}': v for k, v in cand_dict.items()} for cand_dict in cand_preds_dict]

    for idx, colli_eng in enumerate(true_spec_dict.keys()): # TODO: sample
        cand_preds = np.stack([i[colli_eng] for i in cand_preds_dict], axis=0)
        true_spec = true_spec_dict[colli_eng]

        if sparse:
            # sparse_output
            pred_specs = np.zeros((cand_preds.shape[0], true_spec.shape[0]))
            inds = cand_preds[:, :, 0].astype(int)
            pos_1 = np.ones(inds.shape) * np.arange(inds.shape[0])[:, None]
            pred_specs[pos_1.flatten().astype(int), inds.flatten()] = cand_preds[
                :, :, 1
            ].flatten()
        else:
            pred_specs = cand_preds

        if ignore_peak:
            pred_specs[:, int(ignore_peak):] = 0
            true_spec = copy.deepcopy(true_spec)
            true_spec[int(ignore_peak):] = 0

        true_npeaks.append(np.sum(true_spec > 0))

        if func == 'cos':
            norm_pred = norm(pred_specs, axis=-1) + 1e-22  # , ord=2) #+ 1e-22
            norm_true = norm(true_spec, axis=-1) + 1e-22  # , ord=2) #+ 1e-22

            # Cos
            dist.append(1 - np.dot(pred_specs, true_spec) / (norm_pred * norm_true))
        elif func == 'entropy':
            def norm_peaks(prob):
                return prob / (prob.sum(axis=-1, keepdims=True) + 1e-22)

            def entropy(prob):
                # assert np.all(np.abs(prob.sum(axis=-1) - 1) < 5e-3), f"Diff to 1: {np.max(np.abs(prob.sum(axis=-1) - 1))}"
                return -np.sum(prob * np.log(prob + 1e-22), axis=-1)

            norm_pred = norm_peaks(pred_specs)
            norm_true = norm_peaks(true_spec)
            zeros = (pred_specs.sum(axis=-1) == 0)
            entropy_pred = entropy(norm_pred)
            entropy_targ = entropy(norm_true)
            entropy_mix = entropy((norm_pred + norm_true) / 2)
            entropy_dists = (2 * entropy_mix - entropy_pred - entropy_targ) / np.log(4)
            entropy_dists[zeros] = 1 # mask empty spectra to be distance 1
            dist.append(entropy_dists)

    dist = np.array(dist)  # num of colli energy x number of candidates
    # if >=5 peaks: weight=4, elif >=1 peaks: weight=1, else: weight=0
    weights = (np.array(true_npeaks) >= 5) * 3 + (np.array(true_npeaks) >= 1) * 1
    # weights = np.ones(dist.shape[0])
    weights = weights / weights.sum()

    if agg:
        return np.sum(dist * weights[:, None], axis=0)  # number of candidates
    else:
        # return both
        dist = dist[weights > 0] # exclude any objectives that have zero-weights based on filter
        return dist, np.sum(dist * weights[:, None], axis=0)
