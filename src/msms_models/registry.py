"""Capabilities and input assumptions are explicit and independent of a checkpoint."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Model:
    name: str
    task: str
    datasets: tuple[str, ...]
    formula_required: bool
    backend: str
    description: str
    upstream: str

    def to_dict(self):
        return asdict(self)


MODELS = {
    m.name: m
    for m in (
        Model(
            "diffms",
            "de-novo",
            ("msg", "canopus"),
            True,
            "diffms",
            "Graph diffusion conditioned on a mass-spectrum fingerprint",
            "https://github.com/coleygroup/DiffMS",
        ),
        Model(
            "molforge",
            "de-novo",
            ("msg", "canopus"),
            True,
            "molforge",
            "MIST-conditioned molecular sequence decoding",
            "https://arxiv.org/abs/2508.04180",
        ),
        Model(
            "ms-gpt",
            "de-novo",
            ("msg", "canopus"),
            True,
            "ms_gpt",
            "Posterior querying of a conditional molecular language model",
            "https://arxiv.org/abs/2607.23607",
        ),
        Model(
            "frigid-base",
            "de-novo",
            ("msg", "canopus"),
            True,
            "frigid",
            "Fingerprint-conditioned discrete diffusion without refinement",
            "https://arxiv.org/abs/2604.16648",
        ),
        Model(
            "frigid",
            "de-novo",
            ("msg", "canopus"),
            True,
            "frigid",
            "Discrete diffusion with explicit spectral refinement rounds",
            "https://arxiv.org/abs/2604.16648",
        ),
        Model(
            "foam",
            "de-novo",
            ("msg", "canopus"),
            True,
            "foam",
            "Formula-constrained graph optimization with a spectral simulator",
            "https://github.com/coleygroup/foam",
        ),
        Model(
            "subformula-nn",
            "retrieval",
            ("msg",),
            True,
            "subformula",
            "Validation-selected sub-formula nearest-neighbour consensus",
            "https://arxiv.org/abs/2610.02249",
        ),
        Model(
            "mist-noaug",
            "retrieval",
            ("msg",),
            True,
            "mist",
            "MIST trained on real spectra without synthetic spectra",
            "https://github.com/coleygroup/MIST",
        ),
        Model(
            "mist-4m-aug",
            "retrieval",
            ("msg",),
            True,
            "mist",
            "MIST trained with the refined-4M synthetic spectrum collection",
            "https://github.com/coleygroup/MIST",
        ),
        Model(
            "iceberg",
            "retrieval",
            ("msg",),
            False,
            "simulation",
            "Candidate ranking by forward spectrum prediction",
            "https://github.com/coleygroup/ms-pred",
        ),
        Model(
            "glacier",
            "retrieval",
            ("msg",),
            False,
            "simulation",
            "Candidate ranking by GLACIER forward spectrum prediction",
            "https://arxiv.org/abs/2606.29161",
        ),
    )
}

DATASET_ALIASES = {"msg": "msg", "massspecgym": "msg", "canopus": "canopus", "nplib1": "canopus"}


def dataset_name(value: str) -> str:
    try:
        return DATASET_ALIASES[value.lower()]
    except KeyError:
        raise ValueError(f"Unknown dataset {value!r}; choose msg or canopus") from None


def get_model(name: str) -> Model:
    try:
        return MODELS[name]
    except KeyError:
        raise ValueError(f"Unknown model {name!r}; choose from {', '.join(MODELS)}") from None
