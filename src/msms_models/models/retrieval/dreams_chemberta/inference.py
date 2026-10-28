"""Rank candidates by the dot product of mapped DreaMS queries and ChemBERTa CLS embeddings."""

from __future__ import annotations

from ....data.candidates import load_candidates

CHEMBERTA = "Derify/ChemBERTa_augmented_pubchem_13m"


def candidate_pool(values):
    """Drop empty strings and keep the first of duplicates; strings are scored as supplied."""
    return list(dict.fromkeys(v for v in values if v))


def embed_candidates(model, smiles, device, batch_size):
    """CLS embeddings of the sorted unique strings, computed in fixed-size chunks."""
    import torch

    unique = sorted(set(smiles))
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(unique), batch_size):
            chunks.append(model.embed_smiles(unique[start:start + batch_size], device).cpu())
    matrix = torch.cat(chunks) if chunks else torch.empty(0)
    return {s: matrix[i] for i, s in enumerate(unique)}


def embed_queries(model, queries, device, batch_size):
    """Mapped query embeddings `mu`, batched in query order."""
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
    model, _ = build_model(
        assets["checkpoint"],
        chemberta_model=parameters.get("chemberta_model", CHEMBERTA),
        chemberta_revision=parameters.get("chemberta_revision"),
        device=device,
    )
    library = load_candidates(assets["candidates"], queries)
    pools = {q["id"]: candidate_pool(library[q["id"]]) for q in queries}
    embeddings = embed_candidates(
        model, [s for pool in pools.values() for s in pool], device, embedding_batch_size
    )
    query_embeddings = embed_queries(model, queries, device, batch_size)

    rows = []
    with torch.inference_mode():
        for query, mu in zip(queries, query_embeddings, strict=True):
            pool = pools[query["id"]]
            matrix = torch.stack([embeddings[s].to(device) for s in pool])
            scores = mu @ matrix.T
            # Exactly tied scores keep the backend's order.
            order = torch.argsort(scores, descending=True)
            ranked = scores[order]
            rows.append(
                {
                    "id": query["id"],
                    "candidates": [pool[i] for i in order.tolist()],
                    "scores": ranked.double().cpu().tolist(),
                    "status": "ok",
                }
            )
    return rows
