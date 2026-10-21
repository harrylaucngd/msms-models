"""Lightweight capabilities; no model libraries are imported here."""

from .._types import Model

MODELS = (
    Model(
        name="glacier",
        configuration="configs/glacier/{dataset}.yaml",
        task="retrieval",
        datasets=("msg",),
        formula_required=False,
        backend="simulation",
        description="Candidate ranking by GLACIER forward spectrum prediction",
        upstream="https://arxiv.org/abs/2606.29161",
        method="glacier",
        module="msms_models.methods.glacier.inference",
        assets=("candidates", "checkpoint"),
        parameters=(
            "max_heavy_atoms",
            "num_bins",
            "query_intensity_space",
            "sparse_k",
            "upper_limit",
        ),
        dependencies=("torch", "mist", "dgl", "torch_scatter", "ms_pred"),
        uses_mist=False,
    ),
)


def validate(config, queries):
    from ...simulation import validate_metadata

    for query in queries:
        validate_metadata(query)
    if config.parameters.get("query_intensity_space") not in {"native", "raw"}:
        raise ValueError("query_intensity_space must be specified explicitly")
