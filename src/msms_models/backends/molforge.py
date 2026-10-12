"""MIST fingerprint tokens to the native MolForge decoder."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from ..io import atomic_json
from ..posteriors import get_posteriors


class MolForgeDecoder:
    def __init__(
        self,
        checkpoint,
        source_tokenizer,
        target_tokenizer,
        *,
        device="cpu",
        source_length=104,
        target_length=130,
        beam_size=10,
    ):
        import sentencepiece as spm
        import torch

        from .._vendor.molforge import decoder
        from .._vendor.molforge.transformer import Transformer

        self.torch, self.decoder = torch, decoder
        self.device = torch.device(device)
        self.source_length, self.target_length = source_length, target_length
        self.beam_size = beam_size
        self.source_tokenizer = spm.SentencePieceProcessor(model_file=str(source_tokenizer))
        self.target_tokenizer = spm.SentencePieceProcessor(model_file=str(target_tokenizer))
        args = SimpleNamespace(
            src_seq_len=source_length, trg_seq_len=target_length, rank=self.device
        )
        self.model = Transformer(
            src_vocab_size=self.source_tokenizer.GetPieceSize(),
            trg_vocab_size=self.target_tokenizer.GetPieceSize(),
            args=args,
        ).to(self.device)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload.get("model_state_dict", payload)
        clean = {}
        for key, value in state.items():
            normalized = key.removeprefix("module.")
            if normalized in clean:
                raise ValueError(f"Duplicate MolForge checkpoint tensor {normalized}")
            clean[normalized] = value
        self.model.load_state_dict(clean, strict=True)
        self.model.eval()

    def generate(self, on_bits):
        torch = self.torch
        token_ids = self.source_tokenizer.EncodeAsIds(
            " ".join(str(int(b)) for b in sorted(on_bits))
        )
        token_ids = token_ids[: self.source_length]
        token_ids += [0] * (self.source_length - len(token_ids))
        source = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        mask = (source != 0).unsqueeze(1)
        self.decoder.beam_size = self.beam_size
        self.decoder.trg_seq_len = self.target_length
        with torch.inference_mode():
            encoded = self.model.encoder(
                self.model.src_positional_encoder(self.model.src_embedding(source)), mask
            )
            values, scores = self.decoder.beam_search(
                self.model,
                encoded,
                mask,
                self.target_tokenizer,
                self.device,
                return_candidates=True,
            )
        return [str(s).replace(" ", "").strip() for s in values], [float(s) for s in scores]


def predict(config, queries):
    import torch
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    paths = config.require_assets("decoder_checkpoint", "source_tokenizer", "target_tokenizer")
    if "fingerprint_threshold" not in config.parameters:
        raise ValueError("MolForge requires an explicitly specified fingerprint_threshold")
    threshold = float(config.parameters["fingerprint_threshold"])
    if not 0 <= threshold <= 1:
        raise ValueError("Invalid MolForge fingerprint threshold")
    torch.set_num_threads(config.workers)
    torch.manual_seed(config.seed)
    posterior = get_posteriors(config, queries)
    decoder = MolForgeDecoder(
        paths["decoder_checkpoint"],
        paths["source_tokenizer"],
        paths["target_tokenizer"],
        device=config.device,
        beam_size=int(config.parameters.get("beam_size", 10)),
        source_length=int(config.parameters.get("source_length", 104)),
        target_length=int(config.parameters.get("target_length", 130)),
    )
    formula_filter = bool(config.parameters.get("formula_filter", False))
    rows = []
    for query, vector in zip(queries, posterior, strict=True):
        candidates, scores = decoder.generate(np.flatnonzero(vector >= threshold))
        if formula_filter:
            retained = []
            for smiles, score in zip(candidates, scores, strict=True):
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None and rdMolDescriptors.CalcMolFormula(mol) == query["formula"]:
                    retained.append((smiles, score))
            candidates, scores = ([s for s, _ in retained], [s for _, s in retained])
        rows.append({"id": query["id"], "candidates": candidates, "scores": scores, "status": "ok"})
    atomic_json(
        config.output / "decoder_protocol.json",
        {
            "decoder": "MolForge fingerprint-token transformer",
            "fingerprint_threshold": threshold,
            "beam_size": decoder.beam_size,
            "formula_filter": formula_filter,
            "source_length": decoder.source_length,
            "target_length": decoder.target_length,
            "checkpoint_loading": "strict",
            "append_source_eos": False,
        },
    )
    return rows
