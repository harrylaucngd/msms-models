"""MIST retrieval: rank candidates by Tanimoto similarity to the predicted fingerprint."""

from ...data.candidates import load_candidates
from ...encoders.mist import get_posteriors
from .fingerprint import rank_fingerprints


def predict(config, queries):
    """``fingerprint_representation: binary`` thresholds the posterior at ``fingerprint_threshold``."""
    p = config.parameters
    binary = p.get("fingerprint_representation", "raw") == "binary"
    return rank_fingerprints(
        queries,
        get_posteriors(config, queries),
        load_candidates(config.require_assets("candidates")["candidates"], queries),
        threshold=float(p["fingerprint_threshold"]) if binary else None,
        ties=p.get("candidate_ties", "stable"),
    )
