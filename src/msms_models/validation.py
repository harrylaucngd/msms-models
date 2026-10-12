"""Validate portable run assets before importing a model implementation."""

from __future__ import annotations

import json
import subprocess

from .registry import get_model


def required_assets(config):
    model = config.model
    requirements = []
    if model in {"mist-noaug", "mist-4m-aug", "subformula-nn", "iceberg", "glacier"}:
        requirements.append("candidates")
    if model == "subformula-nn":
        requirements.append("nn_index")
    if model in {
        "mist-noaug",
        "mist-4m-aug",
        "molforge",
        "ms-gpt-base",
        "ms-gpt",
        "frigid-base",
        "frigid",
    }:
        requirements += (
            ["posterior", "posterior_ids", "posterior_manifest"]
            if "posterior" in config.assets
            else ["mist_checkpoint", "compact_cache"]
        )
    if model in {"molforge", "ms-gpt-base", "ms-gpt", "frigid-base", "frigid"}:
        requirements.append("decoder_checkpoint")
    if model == "molforge":
        requirements += ["source_tokenizer", "target_tokenizer"]
    if model in {"ms-gpt-base", "ms-gpt"}:
        requirements += ["decoder_config", "decoder_tokenizer", "calibration"]
    if model == "ms-gpt":
        requirements.append("adapter_checkpoint")
    if model in {"frigid-base", "frigid"}:
        requirements += ["safe_tokenizer", "length_model", "length_features"]
    if model in {"iceberg", "frigid", "foam"}:
        requirements += ["generator_checkpoint", "intensity_checkpoint"]
    if model == "foam":
        requirements += ["seeds", "seed_manifest"]
    if model == "glacier":
        requirements.append("checkpoint")
    return requirements


def validate_run(config, queries):
    model = get_model(config.model)
    unknown = set(config.parameters) - PARAMETERS[config.model]
    if unknown:
        raise ValueError(f"Unknown {config.model} parameters: {sorted(unknown)}")
    if config.model == "ms-gpt-base" and "adapter_checkpoint" in config.assets:
        raise ValueError("ms-gpt-base must not receive an adapter_checkpoint; select ms-gpt")
    if model.formula_required and any(not q.get("formula") for q in queries):
        raise ValueError(f"{model.name} requires a known molecular formula for every query")
    config.require_assets(*required_assets(config))
    if config.model == "ms-gpt":
        from .backends.ms_gpt_adapter import validate_adapter_assets

        validate_adapter_assets(config.assets["adapter_checkpoint"])
    p = config.parameters
    if config.model in {"molforge", "frigid-base", "frigid"} and "fingerprint_threshold" not in p:
        raise ValueError("This decoder requires an explicit fingerprint_threshold")
    if config.model in {"iceberg", "glacier", "foam", "frigid"}:
        from .backends.simulation import validate_metadata

        for query in queries:
            validate_metadata(query)
        if p.get("query_intensity_space") not in {"native", "raw"}:
            raise ValueError("query_intensity_space must be specified explicitly")


def doctor(config, queries):
    validate_run(config, queries)
    modules = ["numpy", "scipy", "rdkit"]
    cached_mist = config.model in {"mist-noaug", "mist-4m-aug"} and "posterior" in config.assets
    if config.model != "subformula-nn" and not cached_mist:
        modules += ["torch"]
    if config.model in {
        "mist-noaug",
        "mist-4m-aug",
        "molforge",
        "ms-gpt-base",
        "ms-gpt",
        "frigid-base",
        "frigid",
        "glacier",
    }:
        if "posterior" not in config.assets or config.model == "glacier":
            modules += ["mist"]
    if config.model == "molforge":
        modules += ["sentencepiece"]
    if config.model in {"ms-gpt-base", "ms-gpt", "frigid-base", "frigid"}:
        modules += ["transformers"]
    if config.model in {"ms-gpt-base", "ms-gpt"} and "decoder_tokenizer" in config.assets:
        modules += ["safe"]
    if config.model in {"frigid-base", "frigid"}:
        modules += ["safe", "bionemo", "lightning", "ngboost"]
    if config.model == "ms-gpt":
        modules += ["peft", "safetensors"]
    if config.model in {"iceberg", "glacier", "foam", "frigid"}:
        modules += ["dgl", "torch_scatter"]
    if config.model == "glacier":
        modules += ["ms_pred"]
    if config.model == "foam":
        modules += ["pymoo", "wandb"]
    code = "import importlib.util,json,sys;print(json.dumps({m:importlib.util.find_spec(m) is not None for m in json.loads(sys.argv[1])}))"
    result = subprocess.run(
        [config.python, "-c", code, json.dumps(sorted(set(modules)))],
        capture_output=True,
        text=True,
        check=True,
    )
    dependencies = json.loads(result.stdout)
    return {
        "model": config.model,
        "dataset": config.dataset,
        "queries": len(queries),
        "assets_present": required_assets(config),
        "python": config.python,
        "dependencies": dependencies,
        "ready_for_checkpoint_test": all(dependencies.values()),
        "checkpoint_compatibility_checked": False,
    }


PARAMETERS = {
    "mist-noaug": {
        "fingerprint_representation",
        "fingerprint_threshold",
        "candidate_ties",
        "encoder_batch_size",
    },
    "mist-4m-aug": {
        "fingerprint_representation",
        "fingerprint_threshold",
        "candidate_ties",
        "encoder_batch_size",
    },
    "subformula-nn": {"nn_batch_size", "candidate_ties"},
    "molforge": {
        "fingerprint_threshold",
        "beam_size",
        "source_length",
        "target_length",
        "formula_filter",
        "encoder_batch_size",
    },
    "ms-gpt-base": {
        "num_queries",
        "samples_per_query",
        "max_length",
        "temperature",
        "top_k",
        "top_p",
        "encoder_batch_size",
    },
    "ms-gpt": {
        "num_queries",
        "samples_per_query",
        "max_length",
        "temperature",
        "top_k",
        "top_p",
        "encoder_batch_size",
    },
    "frigid-base": {
        "fingerprint_threshold",
        "num_samples",
        "temperature",
        "randomness",
        "sigma_lambda",
        "encoder_batch_size",
    },
    "frigid": {
        "fingerprint_threshold",
        "query_intensity_space",
        "batch_size",
        "num_rounds",
        "num_unique_to_refine",
        "masks_per_molecule",
        "mask_probability",
        "hallucinated_peaks",
        "temperature",
        "randomness",
        "simulation_energies",
        "simulation_nce",
        "sigma_lambda",
        "max_nodes",
        "sparse_k",
        "iceberg_module",
        "encoder_batch_size",
    },
    "foam": {
        "query_intensity_space",
        "max_calls",
        "population_size",
        "offspring_size",
        "starting_seed_size",
        "num_islands",
        "mutation_rate",
        "num_bins",
        "upper_limit",
        "ignore_precursor_peak",
        "max_nodes",
        "keep_population",
        "truncate",
        "log_frequency",
        "patience",
        "iceberg_module",
    },
    "iceberg": {
        "query_intensity_space",
        "num_bins",
        "upper_limit",
        "sparse_k",
        "fragment_threshold",
        "max_nodes",
        "max_heavy_atoms",
        "iceberg_module",
    },
    "glacier": {"query_intensity_space", "num_bins", "upper_limit", "sparse_k", "max_heavy_atoms"},
}
