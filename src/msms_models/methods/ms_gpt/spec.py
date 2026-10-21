"""Lightweight capabilities; no model libraries are imported here."""

from .._types import Model

MODELS = (
    Model(
        name="ms-gpt-base",
        configuration="configs/ms_gpt/{dataset}-base.yaml",
        task="de-novo",
        datasets=("msg", "canopus"),
        formula_required=True,
        backend="ms_gpt",
        description="Density-band posterior querying without a posterior-aligned adapter",
        upstream="https://arxiv.org/abs/2607.23607",
        method="ms_gpt",
        module="msms_models.methods.ms_gpt.inference",
        assets=("decoder_checkpoint", "decoder_config", "decoder_tokenizer", "calibration"),
        parameters=(
            "encoder_batch_size",
            "max_length",
            "num_queries",
            "samples_per_query",
            "temperature",
            "top_k",
            "top_p",
        ),
        dependencies=("torch", "transformers", "safe"),
        uses_mist=True,
    ),
    Model(
        name="ms-gpt",
        configuration="configs/ms_gpt/{dataset}.yaml",
        task="de-novo",
        datasets=("msg", "canopus"),
        formula_required=True,
        backend="ms_gpt",
        description="Density-band posterior querying with a posterior-aligned LoRA adapter",
        upstream="https://arxiv.org/abs/2607.23607",
        method="ms_gpt",
        module="msms_models.methods.ms_gpt.inference",
        assets=(
            "decoder_checkpoint",
            "decoder_config",
            "decoder_tokenizer",
            "calibration",
            "adapter_checkpoint",
        ),
        parameters=(
            "encoder_batch_size",
            "max_length",
            "num_queries",
            "samples_per_query",
            "temperature",
            "top_k",
            "top_p",
        ),
        dependencies=("torch", "transformers", "safe", "peft", "safetensors"),
        uses_mist=True,
    ),
)


def validate(config, queries):
    if config.model == "ms-gpt-base":
        if "adapter_checkpoint" in config.assets:
            raise ValueError("ms-gpt-base must not receive an adapter_checkpoint; select ms-gpt")
    else:
        from .adapter import validate_adapter_assets

        path = config.require_assets("adapter_checkpoint")["adapter_checkpoint"]
        validate_adapter_assets(path)
