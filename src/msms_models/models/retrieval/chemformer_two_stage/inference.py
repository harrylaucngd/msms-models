"""GLMR two-stage retrieval: contrastive pre-retrieval, then generative re-ranking.

Stage 1 ranks a query's whole candidate pool by the cosine similarity between the
spectrum embedding and each candidate's Chemformer CLS embedding. Stage 2 decodes a
molecule by beam search from the spectrum fused with the top ``pre_retrieval_k``
candidates and re-ranks those candidates by cosine similarity to the decoded
molecule's CLS embedding. The returned list is the re-ranked top K followed by the
rest of the stage-1 order; if decoding yields an empty string the stage-1 order is
kept. Ported from GLMR ``benchmark_massspecgym_full/run_full_with_ci.py``
(``evaluate_full_collect``) with the helpers of ``src/evaluate/eval_full.py``.
"""

from __future__ import annotations

import numpy as np

from ....data.candidates import load_candidates
from ....data.io import atomic_json

DEFAULTS = {
    "max_peaks": 61,
    "pre_retrieval_k": 40,
    "beam_size": 5,
    "max_gen_len": 512,
    "eval_batch_size": 32,
    "candidate_batch_size": 512,
}


def _parameters(config):
    values = {key: int(config.parameters.get(key, default)) for key, default in DEFAULTS.items()}
    for key, value in values.items():
        if value < 1:
            raise ValueError(f"{key} must be positive")
    if values["max_gen_len"] > 512:
        raise ValueError("max_gen_len cannot exceed Chemformer's 512 positions")
    return values


def validate(config, queries):
    _parameters(config)
    empty = [q["id"] for q in queries if not (q.get("spectrum") or {}).get("mz")]
    if empty:
        raise ValueError(f"Chemformer two-stage retrieval needs peaks for every query: {empty[:5]}")


def _autocast(torch, device):
    # GLMR wraps every forward pass in CUDA fp16 autocast, which is inactive on CPU.
    return torch.autocast(device_type="cuda", enabled=device.type == "cuda")


def encode_candidates(model, smiles, tokenizer, device, batch_size=512):
    """L2-normalised CLS embeddings for the sorted unique SMILES, in batches of ``batch_size``.

    The embedding of a string depends only on that string and on which strings share
    its batch (the padded length), so pass the same universe as the original run for
    bit-identical values.
    """
    import torch

    from .data import batch_tokens

    unique = sorted(set(smiles))
    embeddings = {}
    model.eval()
    for start in range(0, len(unique), batch_size):
        chunk = unique[start : start + batch_size]
        ids, mask = batch_tokens([tokenizer.encode(s) for s in chunk])
        ids, mask = ids.to(device), mask.to(device)
        with torch.no_grad(), _autocast(torch, device):
            values = model.encode_molecules_cls(ids, mask).cpu().numpy()
        for j, s in enumerate(chunk):
            embeddings[s] = values[j]
    return embeddings


def rank_two_stage(model, records, pools, embeddings, tokenizer, device, *, max_peaks=61,
                   pre_retrieval_k=40, beam_size=5, max_gen_len=512, eval_batch_size=32):
    """Rank each pool; ``records`` hold float32 ``mzs``/``intensities`` in m/z order.

    Returns one ``(order, scores, reranked, generated)`` tuple per record, where
    ``order`` indexes the pool and ``reranked`` is how many leading entries stage 2
    re-ordered (0 after a generation failure).
    """
    import torch

    from .data import batch_candidates, batch_spectra, batch_tokens, prepare_spectrum

    model.eval()
    results = []
    for start in range(0, len(records), eval_batch_size):
        chunk = list(range(start, min(start + eval_batch_size, len(records))))
        spectra, mask = batch_spectra(
            [prepare_spectrum(records[i]["mzs"], records[i]["intensities"], max_peaks)
             for i in chunk],
            device,
        )
        with torch.no_grad(), _autocast(torch, device):
            spectrum_embeddings = model.encode_spectra_pooled(spectra, mask).cpu().numpy()

        stage1 = []
        for row, i in enumerate(chunk):
            matrix = np.stack([embeddings[s] for s in pools[i]])
            similarity = matrix @ spectrum_embeddings[row]
            # NumPy's default (quicksort) argsort, whose tie order the original keeps.
            stage1.append((np.argsort(-similarity), similarity))
        top = [[pools[i][j] for j in order[:pre_retrieval_k]] for i, (order, _) in
               zip(chunk, stage1)]

        cand_ids, cand_mask = batch_candidates(top, tokenizer, device, k=pre_retrieval_k)
        with torch.no_grad(), _autocast(torch, device):
            generated_ids = model.generate(spectra, mask, cand_ids, cand_mask, max_gen_len,
                                           beam_size)
        generated = [tokenizer.decode(ids) for ids in generated_ids]
        decoded = [(row, s) for row, s in enumerate(generated) if s]
        generated_embeddings = {}
        if decoded:
            ids, gen_mask = batch_tokens(
                [tokenizer.encode(s, max_len=512) for _, s in decoded], device
            )
            with torch.no_grad(), _autocast(torch, device):
                values = model.encode_molecules_cls(ids, gen_mask).cpu().numpy()
            generated_embeddings = {row: values[j] for j, (row, _) in enumerate(decoded)}

        for row, i in enumerate(chunk):
            order, similarity = stage1[row]
            if row not in generated_embeddings:
                results.append((order, similarity[order], 0, generated[row]))
                continue
            head = order[:pre_retrieval_k]
            matrix = np.stack([embeddings[pools[i][j]] for j in head])
            rerank = matrix @ generated_embeddings[row]
            within = np.argsort(-rerank)
            full = np.concatenate([head[within], order[pre_retrieval_k:]])
            scores = np.concatenate([rerank[within], similarity[order[pre_retrieval_k:]]])
            results.append((full, scores, len(head), generated[row]))
    return results


def predict(config, queries):
    import torch

    from .data import query_record
    from .model import build_generative_model
    from .tokenizer import SmilesTokenizer

    paths = config.require_assets(
        "chemformer_checkpoint", "pre_retrieval_checkpoint", "generative_checkpoint",
        "candidates",
    )
    values = _parameters(config)
    device = torch.device(config.device)
    torch.set_num_threads(config.workers)
    library = load_candidates(paths["candidates"], queries)
    pools = [library[q["id"]] for q in queries]
    tokenizer = SmilesTokenizer()
    model = build_generative_model(
        paths["chemformer_checkpoint"],
        paths["pre_retrieval_checkpoint"],
        paths["generative_checkpoint"],
    ).to(device)
    model.eval()
    embeddings = encode_candidates(
        model, (s for pool in pools for s in pool), tokenizer, device,
        batch_size=values["candidate_batch_size"],
    )
    ranked = rank_two_stage(
        model, [query_record(q) for q in queries], pools, embeddings, tokenizer, device,
        **{k: values[k] for k in ("max_peaks", "pre_retrieval_k", "beam_size", "max_gen_len",
                                  "eval_batch_size")},
    )
    rows, generated = [], {}
    for query, pool, (order, scores, reranked, smiles) in zip(queries, pools, ranked,
                                                              strict=True):
        generated[query["id"]] = smiles
        rows.append({
            "id": query["id"],
            "candidates": [pool[j] for j in order],
            # Stage-2 similarity for the first `reranked` entries, stage-1 after them.
            "scores": scores.astype(float).tolist(),
            "reranked": int(reranked),
            "status": "ok",
        })
    atomic_json(config.output / "chemformer_two_stage.json", {
        "parameters": values,
        "unique_candidates_encoded": len(embeddings),
        "generation_failures": sum(not s for s in generated.values()),
        "generated_smiles": generated,
    })
    return rows
