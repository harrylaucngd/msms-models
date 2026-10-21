"""Lightweight capabilities; no model libraries are imported here."""

from .._types import Model

MODELS = (
    Model(
        name="frigid-base",
        configuration="configs/frigid/{dataset}-base.yaml",
        task="de-novo",
        datasets=("msg", "canopus"),
        formula_required=True,
        backend="frigid",
        description="Fingerprint-conditioned discrete diffusion without refinement",
        upstream="https://arxiv.org/abs/2604.16648",
        method="frigid",
        module="msms_models.methods.frigid.inference",
        assets=("decoder_checkpoint", "safe_tokenizer", "length_model", "length_features"),
        parameters=(
            "encoder_batch_size",
            "fingerprint_threshold",
            "num_samples",
            "randomness",
            "sigma_lambda",
            "temperature",
        ),
        dependencies=("torch", "transformers", "safe", "bionemo", "lightning", "ngboost"),
        uses_mist=True,
    ),
    Model(
        name="frigid",
        configuration="configs/frigid/{dataset}.yaml",
        task="de-novo",
        datasets=("msg", "canopus"),
        formula_required=True,
        backend="frigid",
        description="Discrete diffusion with explicit spectral refinement rounds",
        upstream="https://arxiv.org/abs/2604.16648",
        method="frigid",
        module="msms_models.methods.frigid.inference",
        assets=(
            "decoder_checkpoint",
            "safe_tokenizer",
            "length_model",
            "length_features",
            "generator_checkpoint",
            "intensity_checkpoint",
        ),
        parameters=(
            "batch_size",
            "encoder_batch_size",
            "fingerprint_threshold",
            "hallucinated_peaks",
            "iceberg_module",
            "mask_probability",
            "masks_per_molecule",
            "max_nodes",
            "num_rounds",
            "num_unique_to_refine",
            "query_intensity_space",
            "randomness",
            "sigma_lambda",
            "simulation_energies",
            "simulation_nce",
            "sparse_k",
            "temperature",
        ),
        dependencies=(
            "torch",
            "transformers",
            "safe",
            "bionemo",
            "lightning",
            "ngboost",
            "dgl",
            "torch_scatter",
        ),
        uses_mist=True,
    ),
)


def validate(config, queries):
    if "fingerprint_threshold" not in config.parameters:
        raise ValueError("FRIGID requires an explicit fingerprint_threshold")
    if config.model == "frigid":
        from ...simulation import validate_metadata

        for query in queries:
            validate_metadata(query)
        if config.parameters.get("query_intensity_space") != "native":
            raise ValueError("FRIGID refinement expects prepared native spectrum intensities")
