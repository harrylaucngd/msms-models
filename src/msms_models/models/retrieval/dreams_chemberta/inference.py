"""Rank candidates by the dot product of mapped DreaMS queries and ChemBERTa CLS embeddings."""

from __future__ import annotations

from ....data.candidates import load_candidates
from ....data.io import atomic_json

CHEMBERTA = "Derify/ChemBERTa_augmented_pubchem_13m"


def validate(config, queries):
    missing = [q["id"] for q in queries if not q.get("spectrum", {}).get("mz")]
    if missing:
        raise ValueError(f"dreams-chemberta needs a nonempty spectrum for queries {missing[:5]}")
    for name in ("batch_size", "embedding_batch_size"):
        if int(config.parameters.get(name, 1)) < 1:
            raise ValueError(f"{name} must be positive")


def candidate_pool(values):
    """SpecBridge's pool cleaning: drop empty strings, keep the first of duplicates.

    Upstream also passes every string through `_canon_smi`, which returns its input
    unchanged, so candidate strings are scored exactly as supplied.
    """
    return list(dict.fromkeys(v for v in values if v))


def embed_candidates(model, smiles, device, batch_size):
    """CLS embeddings, in upstream order: sorted unique strings in fixed-size chunks."""
    import torch

    unique = sorted(set(smiles))
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(unique), batch_size):
            chunks.append(model.embed_smiles(unique[start:start + batch_size], device).cpu())
    matrix = torch.cat(chunks) if chunks else torch.empty(0)
    return {s: matrix[i] for i, s in enumerate(unique)}


def embed_queries(model, queries, device, batch_size):
    """Mapped query embeddings `mu`, batched in query order as the original evaluation."""
    import torch

    from .model import pad_peaks

    rows = []
    with torch.inference_mode():
        for start in range(0, len(queries), batch_size):
            batch = queries[start:start + batch_size]
            _, mu = model.embed_spectra(pad_peaks([q["spectrum"] for q in batch], device))
            rows.extend(mu)
    return rows


def predict(config, queries):
    import torch

    from .model import build_model

    assets = config.require_assets("checkpoint", "candidates")
    parameters = config.parameters
    batch_size = int(parameters.get("batch_size", 32))
    embedding_batch_size = int(parameters.get("embedding_batch_size", 512))
    device = torch.device(config.device)
    model, training_args = build_model(
        assets["checkpoint"],
        chemberta_model=parameters.get("chemberta_model", CHEMBERTA),
        chemberta_revision=parameters.get("chemberta_revision"),
        device=device,
    )
    library = load_candidates(assets["candidates"], queries)
    pools = {q["id"]: candidate_pool(library[q["id"]]) for q in queries}
    if any(not pool for pool in pools.values()):
        raise ValueError("Every query needs at least one nonempty candidate")
    embeddings = embed_candidates(
        model, [s for pool in pools.values() for s in pool], device, embedding_batch_size
    )
    query_embeddings = embed_queries(model, queries, device, batch_size)

    rows, tied = [], 0
    with torch.inference_mode():
        for query, mu in zip(queries, query_embeddings, strict=True):
            pool = pools[query["id"]]
            matrix = torch.stack([embeddings[s].to(device) for s in pool])
            scores = mu @ matrix.T
            # Upstream call; its order for exactly tied scores is the backend's.
            order = torch.argsort(scores, descending=True)
            ranked = scores[order]
            tied += bool((ranked[1:] == ranked[:-1]).any()) if len(pool) > 1 else 0
            rows.append(
                {
                    "id": query["id"],
                    "candidates": [pool[i] for i in order.tolist()],
                    "scores": ranked.double().cpu().tolist(),
                    "status": "ok",
                }
            )
    config.output.mkdir(parents=True, exist_ok=True)
    atomic_json(
        config.output / "dreams_chemberta.json",
        {
            "queries": len(queries),
            "unique_candidates": len(embeddings),
            "candidates_per_query": {
                "min": min(len(p) for p in pools.values()),
                "max": max(len(p) for p in pools.values()),
            },
            "queries_with_tied_scores": tied,
            "batch_size": batch_size,
            "embedding_batch_size": embedding_batch_size,
            "device": str(device),
            "chemberta_model": parameters.get("chemberta_model", CHEMBERTA),
            "chemberta_revision": parameters.get("chemberta_revision"),
            "training_args": {
                k: training_args.get(k)
                for k in ("mgf", "chemberta_model", "chemberta_ckpt", "epochs", "seed")
            },
            "uses_target_smiles": False,
        },
    )
    return rows
