"""FRIGID generation with explicit tokenizer and formula-only length prediction."""

from __future__ import annotations

import json
import random

import numpy as np

from ....encoders.mist import get_posteriors


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
    # Shared-attention checkpoints store the same cross-attention tensors under both
    # conditioners; the model holds them once, under the formula conditioner.
    if model.use_shared_cross_attention:
        prefix = "fingerprint_conditioner.cross_attention_layers."
        state = {k: v for k, v in state.items() if not k.startswith(prefix)}
    # Two deterministic buffers may be absent from the checkpoint.
    model.load_state_dict(state, strict=False)
    if model.ema:
        model.ema.load_state_dict(checkpoint["ema"])
        model.ema.copy_to(model.backbone.parameters())
    model.to(config.device).eval()
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
    return [{"id": r["name"], "candidates": r["generated_smiles"], "status": "ok"} for r in result]


def measured_spectrum(common, query, *, nce, recorded_energy=True):
    """Raw peaks processed as by ms-pred; the energy is NaN when unknown or not used."""
    energy = query.get("collision_energy", float("nan")) if recorded_energy else float("nan")
    spectrum = common.MassSpec(
        energy,
        masses=np.asarray(query["spectrum"]["mz"], dtype=float),
        intens=np.asarray(query["spectrum"]["intensity"], dtype=float),
    )
    composite = common.CompositeMassSpec([spectrum])
    composite.process_spec_file(parentmass=query["precursor_mz"])
    if nce:
        composite.nce_to_ev(query["precursor_mz"])
    return composite


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

    class PortableRefinement(IcebergSampler):

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
    # "unknown" treats every measured spectrum as having no recorded collision energy.
    recorded = p.get("measured_collision_energy", "recorded") == "recorded"
    for query, vector in zip(queries, posterior, strict=True):
        samples.append(
            {
                "name": query["id"],
                "real_specs": measured_spectrum(common, query, nce=sc.nce, recorded_energy=recorded),
                "target_fp": (vector >= threshold).astype(np.float32),
                "target_formula": query["formula"],
                "instrument": query.get("instrument", "Unknown"),
                "ionization": query["adduct"],
            }
        )
    result = runner.run_scaling_all_samples(samples)
    return [{"id": r["name"], "candidates": r["generated_smiles"], "status": "ok"} for r in result]
