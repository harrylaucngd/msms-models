"""Lightweight capabilities; no model libraries are imported here."""

from .._types import Model

MODELS = (
    Model(
        name="foam",
        configuration="configs/foam/{dataset}.yaml",
        task="de-novo",
        datasets=("msg", "canopus"),
        formula_required=True,
        backend="foam",
        description="Formula-constrained graph optimization with a spectral simulator",
        upstream="https://github.com/coleygroup/foam",
        method="foam",
        module="msms_models.methods.foam.inference",
        assets=("generator_checkpoint", "intensity_checkpoint", "seeds", "seed_manifest"),
        parameters=(
            "iceberg_module",
            "ignore_precursor_peak",
            "keep_population",
            "log_frequency",
            "max_calls",
            "max_nodes",
            "mutation_rate",
            "num_bins",
            "num_islands",
            "offspring_size",
            "patience",
            "population_size",
            "query_intensity_space",
            "starting_seed_size",
            "truncate",
            "upper_limit",
        ),
        dependencies=("torch", "dgl", "torch_scatter", "pymoo", "wandb"),
        uses_mist=False,
    ),
)


def validate(config, queries):
    from ...simulation import validate_metadata

    for query in queries:
        validate_metadata(query)
    if config.parameters.get("query_intensity_space") != "native":
        raise ValueError("FOAM expects prepared native spectrum intensities")
