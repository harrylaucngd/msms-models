"""Lightweight capabilities; no model libraries are imported here."""

from .._types import Model

MODELS = (
    Model(
        name="molforge",
        configuration="configs/molforge/{dataset}.yaml",
        task="de-novo",
        datasets=("msg", "canopus"),
        formula_required=True,
        backend="molforge",
        description="MIST-conditioned molecular sequence decoding",
        upstream="https://arxiv.org/abs/2508.04180",
        method="molforge",
        module="msms_models.methods.molforge.inference",
        assets=("decoder_checkpoint", "source_tokenizer", "target_tokenizer"),
        parameters=(
            "beam_size",
            "encoder_batch_size",
            "fingerprint_threshold",
            "formula_filter",
            "source_length",
            "target_length",
        ),
        dependencies=("torch", "sentencepiece"),
        uses_mist=True,
    ),
)


def validate(config, queries):
    if "fingerprint_threshold" not in config.parameters:
        raise ValueError("MolForge requires an explicit fingerprint_threshold")
