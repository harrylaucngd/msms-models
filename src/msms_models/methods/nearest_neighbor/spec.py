"""Lightweight capabilities; no model libraries are imported here."""

from .._types import Model

MODELS = (
    Model(
        name="subformula-nn",
        configuration="configs/nearest_neighbor/{dataset}.yaml",
        task="retrieval",
        datasets=("msg",),
        formula_required=True,
        backend="subformula",
        description="Validation-selected sub-formula nearest-neighbour consensus",
        upstream="https://arxiv.org/abs/2610.02249",
        method="nearest_neighbor",
        module="msms_models.methods.nearest_neighbor.inference",
        assets=("candidates", "nn_index"),
        parameters=("candidate_ties", "nn_batch_size"),
        dependencies=(),
        uses_mist=False,
    ),
)
