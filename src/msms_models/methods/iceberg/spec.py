"""Lightweight capabilities; no model libraries are imported here."""

from .._types import Model

MODELS = (
    Model(
        name="iceberg",
        configuration="configs/iceberg/{dataset}.yaml",
        task="retrieval",
        datasets=("msg",),
        formula_required=False,
        backend="simulation",
        description="Candidate ranking by forward spectrum prediction",
        upstream="https://github.com/coleygroup/ms-pred",
        method="iceberg",
        module="msms_models.methods.iceberg.inference",
        assets=("candidates", "generator_checkpoint", "intensity_checkpoint"),
        parameters=(
            "fragment_threshold",
            "iceberg_module",
            "max_heavy_atoms",
            "max_nodes",
            "num_bins",
            "query_intensity_space",
            "sparse_k",
            "upper_limit",
        ),
        dependencies=("torch", "dgl", "torch_scatter"),
        uses_mist=False,
    ),
)


def validate(config, queries):
    from ...simulation import validate_metadata

    for query in queries:
        validate_metadata(query)
    if config.parameters.get("query_intensity_space") not in {"native", "raw"}:
        raise ValueError("query_intensity_space must be specified explicitly")
