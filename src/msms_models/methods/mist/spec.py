"""Lightweight capabilities; no model libraries are imported here."""

from .._types import Model

MODELS = (
    Model(
        name="mist-noaug",
        configuration="configs/mist/{dataset}-noaug.yaml",
        task="retrieval",
        datasets=("msg",),
        formula_required=True,
        backend="mist",
        description="MIST trained on real spectra without synthetic spectra",
        upstream="https://github.com/coleygroup/MIST",
        method="mist",
        module="msms_models.methods.mist.inference",
        assets=("candidates",),
        parameters=(
            "candidate_ties",
            "encoder_batch_size",
            "fingerprint_representation",
            "fingerprint_threshold",
        ),
        dependencies=(),
        uses_mist=True,
    ),
    Model(
        name="mist-4m-aug",
        configuration="configs/mist/{dataset}-4m-aug.yaml",
        task="retrieval",
        datasets=("msg",),
        formula_required=True,
        backend="mist",
        description="MIST trained with the refined-4M synthetic spectrum collection",
        upstream="https://github.com/coleygroup/MIST",
        method="mist",
        module="msms_models.methods.mist.inference",
        assets=("candidates",),
        parameters=(
            "candidate_ties",
            "encoder_batch_size",
            "fingerprint_representation",
            "fingerprint_threshold",
        ),
        dependencies=(),
        uses_mist=True,
    ),
)
