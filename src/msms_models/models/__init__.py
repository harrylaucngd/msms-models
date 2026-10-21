"""Registered models. Reading this table imports no model runtime.

Each ``module`` implements ``predict(config, queries)`` returning one ordered
prediction record per query, and optionally ``validate(config, queries)`` for
method-specific input checks. Variants of a model are configurations, not code.
"""

from dataclasses import asdict, dataclass
from importlib import import_module

MIST_POSTERIOR = ("posterior", "posterior_ids", "posterior_manifest")


@dataclass(frozen=True)
class Model:
    name: str
    task: str  # "retrieval" or "de-novo"
    module: str
    datasets: tuple[str, ...] = ("msg",)
    formula_required: bool = True
    assets: tuple[str, ...] = ()
    parameters: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()  # importable runtime modules, checked by `doctor`
    uses_mist: bool = False  # conditioned on MIST fingerprint posteriors

    def to_dict(self):
        return asdict(self)

    def implementation(self):
        return import_module(self.module)

    def required_assets(self, config):
        assets = list(self.assets)
        if self.uses_mist:
            if "posterior" in config.assets:
                assets += MIST_POSTERIOR
            else:
                features = "legacy_features" if "legacy_features" in config.assets else "compact_cache"
                assets += ["mist_checkpoint", features]
        return assets

    def required_modules(self, config):
        modules = {"numpy", "scipy", "rdkit", *self.requires}
        if self.uses_mist and "posterior" not in config.assets:
            modules.add("torch")
            modules.update(["pyarrow", "datamol"] if "legacy_features" in config.assets else ["mist"])
        return sorted(modules)


SPECTRAL = ("max_heavy_atoms", "num_bins", "query_intensity_space", "sparse_k", "upper_limit")
ICEBERG = ("generator_checkpoint", "intensity_checkpoint")
GRAPH = ("torch", "dgl", "torch_scatter")
MIST_DECODER = ("attention_padding", "encoder_batch_size", "fingerprint_threshold")

_MODELS = (
    # Retrieval
    Model(
        "nearest-neighbor", "retrieval", "msms_models.models.retrieval.nearest_neighbor.inference",
        assets=("candidates", "nn_index"), parameters=("candidate_ties", "nn_batch_size"),
    ),
    Model(
        "mist", "retrieval", "msms_models.models.retrieval.mist",
        assets=("candidates",),
        parameters=(
            "attention_padding", "candidate_ties", "encoder_batch_size",
            "fingerprint_representation", "fingerprint_threshold",
        ),
        uses_mist=True,
    ),
    Model(
        "iceberg", "retrieval", "msms_models.models.retrieval.iceberg",
        formula_required=False, assets=("candidates", *ICEBERG),
        parameters=("fragment_threshold", "iceberg_module", "max_nodes", *SPECTRAL),
        requires=GRAPH,
    ),
    Model(
        "glacier", "retrieval", "msms_models.models.retrieval.glacier",
        formula_required=False, assets=("candidates", "checkpoint"), parameters=SPECTRAL,
        requires=(*GRAPH, "mist", "ms_pred"),
    ),
    # De novo generation
    Model(
        "molforge", "de-novo", "msms_models.models.de_novo.molforge.inference",
        datasets=("msg", "canopus"),
        assets=("decoder_checkpoint", "source_tokenizer", "target_tokenizer"),
        parameters=(
            "beam_size", "formula_filter", "source_length", "target_length", *MIST_DECODER,
        ),
        requires=("torch", "sentencepiece"), uses_mist=True,
    ),
    Model(
        "ms-gpt-base", "de-novo", "msms_models.models.de_novo.ms_gpt.inference",
        datasets=("msg", "canopus"),
        assets=("decoder_checkpoint", "decoder_config", "decoder_tokenizer", "calibration"),
        parameters=(
            "attention_padding", "encoder_batch_size", "max_length", "num_queries",
            "samples_per_query", "temperature", "top_k", "top_p",
        ),
        requires=("torch", "transformers", "safe"), uses_mist=True,
    ),
    Model(
        "ms-gpt", "de-novo", "msms_models.models.de_novo.ms_gpt.inference",
        datasets=("msg", "canopus"),
        assets=(
            "decoder_checkpoint", "decoder_config", "decoder_tokenizer", "calibration",
            "adapter_checkpoint",
        ),
        parameters=(
            "attention_padding", "encoder_batch_size", "max_length", "num_queries",
            "samples_per_query", "temperature", "top_k", "top_p",
        ),
        requires=("torch", "transformers", "safe", "peft", "safetensors"), uses_mist=True,
    ),
    Model(
        "frigid-base", "de-novo", "msms_models.models.de_novo.frigid.inference",
        datasets=("msg", "canopus"),
        assets=("decoder_checkpoint", "safe_tokenizer", "length_model", "length_features"),
        parameters=("num_samples", "randomness", "sigma_lambda", "temperature", *MIST_DECODER),
        requires=("torch", "transformers", "safe", "bionemo", "lightning", "ngboost"),
        uses_mist=True,
    ),
    Model(
        "frigid", "de-novo", "msms_models.models.de_novo.frigid.inference",
        datasets=("msg", "canopus"),
        assets=("decoder_checkpoint", "safe_tokenizer", "length_model", "length_features", *ICEBERG),
        parameters=(
            "batch_size", "hallucinated_peaks", "iceberg_module", "mask_probability",
            "masks_per_molecule", "max_nodes", "num_rounds", "num_unique_to_refine",
            "query_intensity_space", "randomness", "sigma_lambda", "simulation_energies",
            "simulation_nce", "sparse_k", "temperature", *MIST_DECODER,
        ),
        requires=("torch", "transformers", "safe", "bionemo", "lightning", "ngboost", *GRAPH[1:]),
        uses_mist=True,
    ),
    Model(
        "foam", "de-novo", "msms_models.models.de_novo.foam.inference",
        datasets=("msg", "canopus"),
        assets=(*ICEBERG, "seeds", "seed_manifest"),
        parameters=(
            "iceberg_module", "ignore_precursor_peak", "keep_population", "log_frequency",
            "max_calls", "max_nodes", "mutation_rate", "num_bins", "num_islands",
            "offspring_size", "patience", "population_size", "query_intensity_space",
            "starting_seed_size", "truncate", "upper_limit",
        ),
        requires=(*GRAPH, "pymoo", "wandb"),
    ),
)

MODELS: dict[str, Model] = {model.name: model for model in _MODELS}
assert len(MODELS) == len(_MODELS), "Duplicate model registration"


def get_model(name: str) -> Model:
    try:
        return MODELS[name]
    except KeyError:
        raise ValueError(f"Unknown model {name!r}; choose from {', '.join(MODELS)}") from None
