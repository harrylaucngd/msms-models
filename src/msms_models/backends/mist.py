"""MIST fingerprint retrieval with explicit representation and threshold settings."""

from ..posteriors import get_posteriors
from ..retrieval.fingerprints import load_candidates, rank_fingerprints


def predict(config, queries):
    assets = config.require_assets("candidates")
    probabilities = get_posteriors(config, queries)
    representation = config.parameters.get("fingerprint_representation", "raw")
    if representation not in {"raw", "binary"}:
        raise ValueError("fingerprint_representation must be raw or binary")
    threshold = None
    if representation == "binary":
        if "fingerprint_threshold" not in config.parameters:
            raise ValueError(
                "Binary retrieval requires an explicitly calibrated fingerprint_threshold"
            )
        threshold = float(config.parameters["fingerprint_threshold"])
    return rank_fingerprints(
        queries,
        probabilities,
        load_candidates(assets["candidates"], queries),
        threshold=threshold,
        ties=config.parameters.get("candidate_ties", "stable"),
    )
