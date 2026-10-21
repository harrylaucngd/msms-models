"""Sub-formula neighbour consensus, then fingerprint-based candidate retrieval."""

from ....data.candidates import load_candidates
from ..fingerprint import rank_fingerprints
from .index import SubformulaIndex


def predict(config, queries):
    paths = config.require_assets("nn_index", "candidates")
    model = SubformulaIndex.load(paths["nn_index"])
    probabilities, _ = model.predict(queries, batch_size=int(config.parameters.get("nn_batch_size", 16)))
    return rank_fingerprints(
        queries, probabilities, load_candidates(paths["candidates"], queries),
        ties=config.parameters.get("candidate_ties", "stable"),
    )
