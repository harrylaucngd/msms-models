"""MS-GPT SAFE decoder queried with ScaledMIST posteriors in a training-calibrated density band."""

from __future__ import annotations

import json
import re
from contextlib import nullcontext

import numpy as np
import yaml

from ....encoders.mist import get_posteriors

ELEMENTS = ("C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B", "Si", "Se", "As")


def formula_vector(formula):
    pieces = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
    counts = {}
    for element, count in pieces:
        counts[element] = counts.get(element, 0) + int(count or 1)
    unsupported = set(counts) - set(ELEMENTS)
    if unsupported:
        raise ValueError(f"MS-GPT formula vocabulary does not contain {sorted(unsupported)}")
    return np.array([counts.get(element, 0) for element in ELEMENTS], np.float32)


def load_decoder(checkpoint_path, model_config, device, tokenizer_path=None):
    import torch
    from safe.tokenizer import SAFETokenizer

    from .vendor.model import (
        ConditionalMoleculeLanguageModel,
        normalize_mlm_checkpoint_state_dict,
    )

    architecture = yaml.safe_load(model_config.read_text())
    tokenizer = SAFETokenizer.from_pretrained(
        str(tokenizer_path), local_files_only=True
    ).get_pretrained()
    tokenizer.add_tokens(["<", ">"])
    config = architecture["model"]
    model = ConditionalMoleculeLanguageModel(
        vocab_size=len(tokenizer),
        dropout=0.0,
        **{
            key: config[key]
            for key in (
                "d_model",
                "n_layers",
                "n_heads",
                "fp_bits",
                "fp_encoder_layers",
                "cross_attn_every_n",
                "max_seq_len",
            )
        },
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload.get("model", payload))
    model.load_state_dict(normalize_mlm_checkpoint_state_dict(state), strict=True)
    return model.to(device).eval(), tokenizer, config


def decode_tokens(tokenizer, tokens):
    if tokenizer.eos_token_id in tokens:
        tokens = tokens[: tokens.index(tokenizer.eos_token_id)]
    import safe

    text = tokenizer.decode(tokens, skip_special_tokens=True)
    try:
        smiles = safe.decode(text)
        if smiles:
            return smiles
    except Exception:
        pass
    return text


def rank_generated(smiles, posterior, formula, band):
    from .vendor.ranking import rank_spectrum_candidates

    # The native function also returns optional hit diagnostics. An empty target
    # key makes clear that target structures play no role in candidate ordering.
    record = {
        "candidate_smiles": smiles,
        "target_formula": formula,
        "target_inchikey14": "",
        "cache_row": 0,
    }
    result = rank_spectrum_candidates((record, True, posterior, band))
    return [result["ik_to_smi"][key] for key in result["ranked_frequency_posterior_similarity"]]


def predict(config, queries):
    import torch

    from .vendor.queries import density_calibrated_query_band

    paths = config.require_assets(
        "decoder_checkpoint", "decoder_config", "decoder_tokenizer", "calibration"
    )
    # The density band calibrated on the training posteriors of the encoder in use.
    calibration = json.loads(paths["calibration"].read_text())
    band = (float(calibration["threshold_low"]), float(calibration["threshold_high"]))
    torch.set_num_threads(config.workers)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device(config.device)
    posterior = get_posteriors(config, queries)
    model, tokenizer, model_config = load_decoder(
        paths["decoder_checkpoint"],
        paths["decoder_config"],
        device,
        tokenizer_path=config.assets.get("decoder_tokenizer"),
    )
    nq = int(config.parameters.get("num_queries", 20))
    ns = int(config.parameters.get("samples_per_query", 5))
    rows = []
    for query, vector in zip(queries, posterior, strict=True):
        try:
            formula = formula_vector(query["formula"])
        except ValueError as error:
            rows.append(
                {
                    "id": query["id"],
                    "candidates": [],
                    "status": "unsupported_formula",
                    "reason": str(error),
                }
            )
            continue
        binary = density_calibrated_query_band(vector, nq, *band)
        fp = torch.tensor(binary, dtype=torch.float32, device=device)
        fv = torch.tensor(np.repeat(formula[None], nq, axis=0), device=device)
        context = (
            torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
        )
        with torch.inference_mode(), context:
            generated = model.generate(
                fingerprint=fp,
                formula_vec=fv,
                max_len=int(config.parameters.get("max_length", model_config["max_seq_len"])),
                temperature=float(config.parameters.get("temperature", 1.0)),
                top_k=int(config.parameters.get("top_k", 50)),
                top_p=float(config.parameters.get("top_p", 0.95)),
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_samples=ns,
            )
        strings = []
        for tokens in generated.cpu().tolist():
            strings.append(decode_tokens(tokenizer, tokens))
        candidates = rank_generated(strings, vector, query["formula"], band)
        rows.append(
            {
                "id": query["id"],
                "candidates": candidates,
                "status": "ok",
                "generated_count": len(strings),
            }
        )
    return rows
