"""FRIGID generation with explicit tokenizer and formula-only length prediction."""

from __future__ import annotations

import json
import random

import numpy as np

from ....data.io import atomic_json
from ....encoders.mist import get_posteriors


def validate(config, queries):
    threshold = config.parameters.get("fingerprint_threshold")
    if threshold is None or not 0 <= float(threshold) <= 1:
        raise ValueError("FRIGID requires an explicit, training-calibrated fingerprint_threshold")
    if config.model == "frigid":
        from ....simulators.retrieve import validate_metadata

        for query in queries:
            validate_metadata(query)
        if config.parameters.get("query_intensity_space") != "native":
            raise ValueError("FRIGID refinement expects prepared native spectrum intensities")


def load_sampler(config):
    import torch

    from .vendor.model import DLM
    from .vendor.sampler import Sampler
    from .vendor.utils.utils_chem import Slicer

    paths = config.require_assets("decoder_checkpoint", "safe_tokenizer")
    checkpoint = torch.load(
        paths["decoder_checkpoint"], map_location="cpu", weights_only=False, mmap=True
    )
    model = DLM(checkpoint["hyper_parameters"]["config"], tokenizer_path=paths["safe_tokenizer"])
    state = dict(checkpoint["state_dict"])
    # Historical shared-attention checkpoints store the same learned tensors
    # under both conditioners. Remove only verified exact aliases.
    if model.use_shared_cross_attention:
        for key in list(state):
            prefix = "fingerprint_conditioner.cross_attention_layers."
            if key.startswith(prefix):
                canonical = key.replace(prefix, "formula_conditioner.cross_attention_layers.", 1)
                if canonical not in state or not torch.equal(state[key], state[canonical]):
                    raise ValueError(f"FRIGID shared-attention alias differs from {canonical}")
                del state[key]
    # DLM's upstream loader permits two deterministic buffers to be absent.
    # Audit all returned keys because older loaders also ignored unexpected ones.
    result = model.load_state_dict(state, strict=True)
    allowed = {"token_atom_counts", "formula_loss_atom_weights"}
    if set(result.missing_keys) - allowed or result.unexpected_keys:
        raise ValueError(f"Incompatible FRIGID weights: {result}")
    if model.ema:
        model.ema.load_state_dict(checkpoint["ema"])
        model.ema.copy_to(model.backbone.parameters())
    model.to(config.device).eval()
    if not model.use_formula_conditioning or not model.use_fingerprint_conditioning:
        raise ValueError("FRIGID checkpoint must support both formula and fingerprint conditioning")
    sampler = Sampler.__new__(Sampler)
    sampler.model = model
    sampler.slicer = Slicer()
    sampler.dot_index = model.tokenizer(".")["input_ids"][1]
    sampler.pad_index = model.tokenizer.pad_token_id
    sampler.mdlm = model.mdlm
    sampler.mdlm.to_device(model.device)
    sampler.fingerprint_bits = model.config.model.get("fingerprint_bits", 4096)
    sampler.max_position_embeddings = model.config.model.get("max_position_embeddings", 256)
    sampler.seq_len_list = list(range(20, 100))
    sampler.token_count_model = None
    sampler.token_count_features = None
    return sampler


def load_length_model(config):
    import joblib

    from .vendor.utils.benchmark_utils import is_ngboost_model

    paths = config.require_assets("length_model", "length_features")
    model = joblib.load(paths["length_model"])
    features = json.loads(paths["length_features"].read_text())
    if not isinstance(features, list) or not features:
        raise ValueError("length_features must be a nonempty list in training column order")
    return model, features, is_ngboost_model(model)


def predict(config, queries):
    import torch

    torch.set_num_threads(config.workers)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    posterior = get_posteriors(config, queries)
    sampler = load_sampler(config)
    length_model, features, ngboost = load_length_model(config)
    if config.model == "frigid":
        return refine(config, queries, posterior, sampler, length_model, features, ngboost)
    return base_round(config, queries, posterior, sampler, length_model, features, ngboost)


def base_round(config, queries, posterior, sampler, length_model, features, ngboost):
    from .vendor.iceberg_sampler import (
        IcebergConfig,
        IcebergSampler,
        ScalingConfig,
    )

    p = config.parameters
    budget = int(p.get("num_samples", 128))
    if budget < 1:
        raise ValueError("FRIGID-base num_samples must be positive")
    scaling = ScalingConfig(
        batch_size=budget,
        num_unique_to_refine=1,
        masks_per_molecule=1,
        num_rounds=1,
        softmax_temp=float(p.get("temperature", 1.0)),
        randomness=float(p.get("randomness", 10.0)),
        seed=config.seed,
        defer_next_round_iceberg=True,
    )
    runner = IcebergSampler(
        sampler,
        IcebergConfig(),
        scaling,
        results_dir=str(config.output / "generation"),
        token_model=length_model,
        token_features=features,
        is_ngboost=ngboost,
        sigma_lambda=float(p.get("sigma_lambda", 3.0)),
    )
    runner.reraise_generation_errors = True
    threshold = float(p["fingerprint_threshold"])
    samples = [
        {
            "name": q["id"],
            "real_specs": None,
            "target_fp": (v >= threshold).astype(np.float32),
            "target_formula": q["formula"],
        }
        for q, v in zip(queries, posterior, strict=True)
    ]
    result = runner.run_scaling_all_samples(samples)
    atomic_json(
        config.output / "decoder_protocol.json",
        {
            "decoder": "FRIGID-base",
            "protocol": "first native refinement round",
            "scratch_proposals": budget,
            "spectral_prediction_calls": 0,
            "fingerprint_threshold": threshold,
            "length_prediction": "formula only",
            "ranking": "native formula priority and fingerprint Tanimoto",
        },
    )
    return [{"id": r["name"], "candidates": r["generated_smiles"], "status": "ok"} for r in result]


def refine(config, queries, posterior, sampler, length_model, features, ngboost):
    import torch

    from ....simulators.iceberg import load_model
    from ....simulators.iceberg.vendor import common
    from .vendor.iceberg_sampler import (
        IcebergConfig,
        IcebergSampler,
        ScalingConfig,
        is_valid_smiles,
    )

    p = config.parameters
    if not hasattr(common, "CompositeMassSpec"):
        raise ValueError(
            "FRIGID refinement requires the ms-pred version specified in its environment"
        )
    forward_model = load_model(config)
    paths = config.require_assets("generator_checkpoint", "intensity_checkpoint")
    ic = IcebergConfig(
        gen_ckpt=str(paths["generator_checkpoint"]),
        inten_ckpt=str(paths["intensity_checkpoint"]),
        num_cpu_workers=config.workers,
        num_gpu_workers=1,
        max_nodes=int(p.get("max_nodes", 100)),
        sparse_k=int(p.get("sparse_k", 100)),
    )
    sc = ScalingConfig(
        batch_size=int(p.get("batch_size", 128)),
        num_unique_to_refine=int(p.get("num_unique_to_refine", 8)),
        masks_per_molecule=int(p.get("masks_per_molecule", 12)),
        num_rounds=int(p.get("num_rounds", 11)),
        mask_prob=float(p.get("mask_probability", 0.75)),
        top_k_halluc_peaks=int(p.get("hallucinated_peaks", 10)),
        softmax_temp=float(p.get("temperature", 1.0)),
        randomness=float(p.get("randomness", 10.0)),
        collision_energies=list(p.get("simulation_energies", range(10, 100, 10))),
        nce=bool(p.get("simulation_nce", True)),
        seed=config.seed,
    )
    if sc.num_rounds < 2:
        raise ValueError(
            "FRIGID refinement requires at least two rounds; use frigid-base for base generation"
        )

    class PortableRefinement(IcebergSampler):
        reraise_generation_errors = True
        prediction_calls = 0

        def _run_iceberg_batch(
            self, smiles_list, collision_energies, instrument="Unknown", ionization="[M+H]+"
        ):
            result = {}
            for smiles in smiles_list:
                if not is_valid_smiles(smiles):
                    continue
                precursor = common.mass_from_smi(smiles) + common.ion2mass[ionization]
                spectra = []
                for energy in collision_energies:
                    ev = common.nce_to_ev(energy, precursor) if sc.nce else energy
                    with torch.inference_mode():
                        output = forward_model.predict_mol(
                            smiles,
                            ev,
                            precursor,
                            ionization,
                            ic.threshold,
                            config.device,
                            ic.max_nodes,
                            instrument=instrument,
                            adduct_shift=True,
                            canonical_root_smi=True,
                            binned_out=False,
                        )
                    raw, fragments = output["spec"], output["frag"]
                    raw = (
                        raw.detach().cpu().numpy()
                        if isinstance(raw, torch.Tensor)
                        else np.asarray(raw)
                    )
                    fragments = (
                        fragments.detach().cpu().numpy()
                        if isinstance(fragments, torch.Tensor)
                        else np.asarray(fragments)
                    )
                    if not np.isfinite(raw).all():
                        raise ValueError("Non-finite refinement spectrum")
                    order = np.argsort(raw[:, 1])[::-1][: ic.sparse_k]
                    fragment_args = (
                        {"frags": fragments[order]}
                        if fragments.ndim == 2
                        else {"int_frags": fragments[order]}
                    )
                    spectra.append(
                        common.MassSpec(
                            collision_energy=ev,
                            root_canonical_smiles=smiles,
                            adduct=ionization,
                            masses=raw[order, 0],
                            intens=raw[order, 1],
                            **fragment_args,
                        )
                    )
                    self.prediction_calls += 1
                result[smiles] = common.CompositeMassSpec(spectra)
            return result

    runner = PortableRefinement(
        sampler,
        ic,
        sc,
        results_dir=str(config.output / "refinement"),
        token_model=length_model,
        token_features=features,
        is_ngboost=ngboost,
        sigma_lambda=float(p.get("sigma_lambda", 3.0)),
        incl_unknown_instrument=False,
    )
    samples = []
    threshold = float(p["fingerprint_threshold"])
    for query, vector in zip(queries, posterior, strict=True):
        spectrum = common.MassSpec(
            collision_energy=query["collision_energy"],
            masses=np.asarray(query["spectrum"]["mz"], dtype=float),
            intens=np.asarray(query["spectrum"]["intensity"], dtype=float),
        )
        samples.append(
            {
                "name": query["id"],
                "real_specs": common.CompositeMassSpec([spectrum]),
                "target_fp": (vector >= threshold).astype(np.float32),
                "target_formula": query["formula"],
                "instrument": query["instrument"],
                "ionization": query["adduct"],
            }
        )
    result = runner.run_scaling_all_samples(samples)
    atomic_json(
        config.output / "decoder_protocol.json",
        {
            "decoder": "FRIGID",
            "rounds": sc.num_rounds,
            "batch_size": sc.batch_size,
            "num_unique_to_refine": sc.num_unique_to_refine,
            "masks_per_molecule": sc.masks_per_molecule,
            "mask_probability": sc.mask_prob,
            "forward_prediction_calls": runner.prediction_calls,
            "simulation_energies": sc.collision_energies,
            "simulation_nce": sc.nce,
            "fingerprint_threshold": threshold,
            "length_prediction": "formula only",
            "ranking": "native accumulated unique candidates by fingerprint Tanimoto",
        },
    )
    return [{"id": r["name"], "candidates": r["generated_smiles"], "status": "ok"} for r in result]
