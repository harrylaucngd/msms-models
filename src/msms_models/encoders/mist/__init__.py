"""MIST fingerprint posteriors for retrieval and fingerprint-conditioned generation.

``get_posteriors`` reads a posterior cache, or encodes spectra with ScaledMIST
(``scaled.py``) or, for the paper's vDiffMS diagnostic only, the MIST released with
DiffMS (``diffms.py``).
"""

from __future__ import annotations

import numpy as np


def align_posterior(values, ids, queries):
    """Rows of ``values`` in query order, matched by spectrum ID."""
    index = {str(identifier): i for i, identifier in enumerate(ids)}
    return np.asarray(values)[[index[q["id"]] for q in queries]].astype(np.float32)


def save_posteriors(config, queries, values):
    """Write the posteriors of a run as a reusable cache in ``<output>/posterior``."""
    output = config.output / "posterior"
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "probabilities.npy", values)
    np.save(output / "query_ids.npy", np.array([q["id"] for q in queries]))
    return values


def get_posteriors(config, queries: list[dict]) -> np.ndarray:
    if "posterior" in config.assets:
        paths = config.require_assets("posterior", "posterior_ids")
        return align_posterior(np.load(paths["posterior"], mmap_mode="r"),
                               np.load(paths["posterior_ids"]), queries)
    if "diffms_features" in config.assets:
        from .diffms import encode
    else:
        from .scaled import encode
    return save_posteriors(config, queries, encode(config, queries))
