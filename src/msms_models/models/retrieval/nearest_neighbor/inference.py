"""Sub-formula consensus followed by fingerprint-based candidate retrieval."""

import numpy as np

from ....data.candidates import load_candidates
from ....data.io import atomic_json
from ..fingerprint import rank_fingerprints
from .index import SubformulaIndex


def predict(config, queries):
    paths = config.require_assets("nn_index", "candidates")
    model = SubformulaIndex.load(paths["nn_index"])
    if model.metadata.get("dataset") != config.dataset:
        raise ValueError("The sub-formula training index belongs to a different dataset")
    probabilities, details = model.predict(
        queries, batch_size=int(config.parameters.get("nn_batch_size", 16))
    )
    config.output.mkdir(parents=True, exist_ok=True)
    np.save(config.output / "fingerprints.npy", probabilities)
    np.save(config.output / "query_ids.npy", np.array([q["id"] for q in queries]))
    np.save(config.output / "neighbour_indices.npy", details["neighbours"])
    np.save(config.output / "neighbour_weights.npy", details["weights"])
    atomic_json(
        config.output / "neighbour_summary.json",
        {
            "parameters": model.parameters,
            "training_rows": len(model.ids),
            "empty_token_rows": details["empty_token_rows"],
            "formula_filter": False,
            "formula_known": True,
            "queries": len(queries),
        },
    )
    del model
    return rank_fingerprints(
        queries,
        probabilities,
        load_candidates(paths["candidates"], queries),
        ties=config.parameters.get("candidate_ties", "stable"),
    )
