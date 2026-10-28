"""GLMR networks: Chemformer encoder/decoder, spectral encoder and cross-fusion.

Module and parameter names match the released checkpoints, which load strictly.
Construction order matters for training: building a module consumes the global torch RNG.
"""

from __future__ import annotations

import math
import pickle
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenizer import END_IDX, PAD_IDX, START_IDX

# Spectral encoder, cross-fusion and molecular encoder sizes of every GLMR checkpoint.
SPECTRAL_DEFAULTS = {
    "spec_d_model": 512,
    "spec_num_layers": 6,
    "spec_num_heads": 8,
    "spec_d_feedforward": 2048,
    "spec_dropout": 0.1,
    "activation": "gelu",
}


class _Opaque:
    """Placeholder for Lightning/DeepSpeed objects pickled into the Chemformer file."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        self.__dict__["state"] = state


class _ChemformerUnpickler(pickle.Unpickler):
    # The 2021 Chemformer checkpoint pickles a Lightning 1.2.3 ModelCheckpoint callback
    # and a DeepSpeed loss scaler next to its weights. Neither is needed, so they are
    # replaced by placeholders instead of importing those frameworks.
    def find_class(self, module, name):
        if module.split(".")[0] in {"pytorch_lightning", "lightning", "deepspeed"}:
            return type(name, (_Opaque,), {})
        return super().find_class(module, name)


_PICKLE = SimpleNamespace(Unpickler=_ChemformerUnpickler, load=pickle.load, __name__="pickle")


def load_chemformer_checkpoint(path) -> tuple[dict, dict]:
    """Return ``(hyper_parameters, state_dict)`` of a Chemformer BART checkpoint."""
    payload = torch.load(path, map_location="cpu", weights_only=False, pickle_module=_PICKLE)
    return dict(payload["hyper_parameters"]), payload["state_dict"]


def _architecture(hp: dict) -> dict:
    return {
        "vocab_size": hp["vocab_size"],
        "d_model": hp["d_model"],
        "num_layers": hp["num_layers"],
        "num_heads": hp["num_heads"],
        "d_feedforward": hp["d_feedforward"],
        "dropout": hp.get("dropout", 0.1),
        "activation": hp.get("activation", "gelu"),
        "max_seq_len": hp.get("max_seq_len", 512),
        "pad_token_idx": hp.get("pad_token_idx", 0),
    }


class ChemformerEncoder(nn.Module):
    """Pre-norm BART encoder; the CLS embedding is the start token's hidden state."""

    def __init__(
        self,
        vocab_size=523,
        d_model=512,
        num_layers=6,
        num_heads=8,
        d_feedforward=2048,
        dropout=0.1,
        activation="gelu",
        max_seq_len=512,
        pad_token_idx=PAD_IDX,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_token_idx = pad_token_idx
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(max_seq_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=False,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, norm=nn.LayerNorm(d_model))

    def forward_sequence(self, token_ids, attention_mask=None):
        """Hidden states ``(B, S, d)``; ``attention_mask`` is True for real tokens."""
        x = self.emb(token_ids) * math.sqrt(self.d_model)
        x = x + self.pos_emb[: token_ids.shape[1]].unsqueeze(0)
        if attention_mask is not None:
            padding = ~attention_mask
        else:
            padding = token_ids == self.pad_token_idx
        x = self.encoder(x.transpose(0, 1), src_key_padding_mask=padding)
        return x.transpose(0, 1)

    def forward(self, token_ids, attention_mask=None):
        return self.forward_sequence(token_ids, attention_mask)[:, 0, :]

    @classmethod
    def from_chemformer(cls, hp: dict, state_dict: dict):
        model = cls(**_architecture(hp))
        own = {
            k: v
            for k, v in state_dict.items()
            if k in {"emb.weight", "pos_emb"} or k.startswith("encoder.")
        }
        model.load_state_dict(own, strict=True)
        return model


class ChemformerDecoder(nn.Module):
    """Pre-norm BART decoder with GLMR's batched beam search."""

    def __init__(
        self,
        vocab_size=523,
        d_model=512,
        num_layers=6,
        num_heads=8,
        d_feedforward=2048,
        dropout=0.1,
        activation="gelu",
        max_seq_len=512,
        pad_token_idx=PAD_IDX,
        start_token_idx=START_IDX,
        end_token_idx=END_IDX,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.pad_token_idx = pad_token_idx
        self.start_token_idx = start_token_idx
        self.end_token_idx = end_token_idx
        self.max_seq_len = max_seq_len
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(max_seq_len, d_model))
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=False,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers, norm=nn.LayerNorm(d_model))
        self.token_fc = nn.Linear(d_model, vocab_size)

    def _embed(self, tokens):
        x = self.emb(tokens) * math.sqrt(self.d_model)
        x = x + self.pos_emb[: tokens.shape[1]].unsqueeze(0)
        causal = nn.Transformer.generate_square_subsequent_mask(
            tokens.shape[1], device=x.device, dtype=x.dtype
        )
        return x, causal

    def forward(self, tgt_ids, memory, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """Teacher-forced logits ``(B, T, V)``; masks are True for real positions."""
        x, causal = self._embed(tgt_ids)
        tgt_pad = None if tgt_key_padding_mask is None else ~tgt_key_padding_mask
        mem_pad = None if memory_key_padding_mask is None else ~memory_key_padding_mask
        out = self.decoder(
            x.transpose(0, 1),
            memory.transpose(0, 1),
            tgt_mask=causal,
            tgt_key_padding_mask=tgt_pad,
            memory_key_padding_mask=mem_pad,
        )
        return self.token_fc(out.transpose(0, 1))

    def _decode_step(self, tokens, memory, memory_mask):
        x, causal = self._embed(tokens)
        mem_pad = None if memory_mask is None else ~memory_mask
        out = self.decoder(
            x.transpose(0, 1),
            memory.transpose(0, 1),
            tgt_mask=causal,
            memory_key_padding_mask=mem_pad,
        )
        return self.token_fc(out.transpose(0, 1))

    @torch.no_grad()
    def generate(self, memory, memory_key_padding_mask=None, max_len=512, beam_size=5):
        """Best beam per sample as token ids, without the start and end tokens."""
        B, S, d = memory.shape
        device = memory.device
        V = self.vocab_size
        total = B * beam_size
        mem = memory.unsqueeze(1).expand(-1, beam_size, -1, -1).reshape(total, S, d)
        mem_mask = None
        if memory_key_padding_mask is not None:
            mem_mask = (
                memory_key_padding_mask.unsqueeze(1).expand(-1, beam_size, -1).reshape(total, S)
            )
        tokens = torch.full((total, 1), self.start_token_idx, dtype=torch.long, device=device)
        scores = torch.full((total,), -1e9, device=device)
        scores[::beam_size] = 0.0  # only the first beam of each sample starts alive
        finished = torch.zeros(total, dtype=torch.bool, device=device)
        for _ in range(max_len - 1):
            if finished.all():
                break
            logits = self._decode_step(tokens, mem, mem_mask)
            log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
            if finished.any():
                # Finished beams may only extend with padding, at no cost.
                log_probs[finished] = -1e9
                log_probs[finished, self.pad_token_idx] = 0.0
            flat = (scores.unsqueeze(-1) + log_probs).view(B, beam_size * V)
            top_scores, top_flat = flat.topk(beam_size, dim=-1)
            beam_idx = top_flat // V
            tok_idx = top_flat % V
            offsets = torch.arange(B, device=device).unsqueeze(1) * beam_size
            rows = (offsets + beam_idx).reshape(-1)
            tokens = torch.cat([tokens[rows], tok_idx.reshape(-1, 1)], dim=-1)
            scores = top_scores.reshape(-1)
            finished = finished[rows] | (tok_idx.reshape(-1) == self.end_token_idx)
        best = scores.view(B, beam_size).argmax(dim=-1)
        sequences = tokens.view(B, beam_size, -1)
        results = []
        for b in range(B):
            seq = sequences[b, best[b]].tolist()
            if self.start_token_idx in seq:
                seq = seq[seq.index(self.start_token_idx) + 1 :]
            if self.end_token_idx in seq:
                seq = seq[: seq.index(self.end_token_idx)]
            results.append(seq)
        return results

    @classmethod
    def from_chemformer(cls, hp: dict, state_dict: dict, encoder: ChemformerEncoder | None = None):
        model = cls(**_architecture(hp))
        own = {
            k: v
            for k, v in state_dict.items()
            if k in {"emb.weight", "pos_emb"} or k.startswith(("decoder.", "token_fc."))
        }
        model.load_state_dict(own, strict=True)
        if encoder is not None:
            # GLMR ties the token and position embeddings to the (frozen) encoder.
            model.emb = encoder.emb
            model.pos_emb = encoder.pos_emb
        return model


class SpectralEncoder(nn.Module):
    """Transformer over (m/z, intensity) peaks; mean pooling over real peaks."""

    def __init__(
        self,
        d_model=512,
        num_layers=6,
        num_heads=8,
        d_feedforward=2048,
        dropout=0.1,
        activation="gelu",
    ):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(2, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=False,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, norm=nn.LayerNorm(d_model))

    def forward_sequence(self, spectra, mask=None):
        x = self.input_proj(spectra)
        padding = None if mask is None else ~mask
        x = self.encoder(x.transpose(0, 1), src_key_padding_mask=padding)
        return x.transpose(0, 1)

    def forward(self, spectra, mask=None):
        x = self.forward_sequence(spectra, mask)
        if mask is None:
            return x.mean(dim=1)
        weights = mask.unsqueeze(-1).float()
        return (x * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)

    @classmethod
    def from_config(cls, model_cfg: dict):
        cfg = {**SPECTRAL_DEFAULTS, **model_cfg}
        return cls(
            d_model=cfg["spec_d_model"],
            num_layers=cfg["spec_num_layers"],
            num_heads=cfg["spec_num_heads"],
            d_feedforward=cfg["spec_d_feedforward"],
            dropout=cfg["spec_dropout"],
            activation=cfg["activation"],
        )


class CrossFusion(nn.Module):
    """``LayerNorm(Hs + CrossAttn(Q=Hs, K=V=Hm))`` over the K concatenated candidates."""

    def __init__(self, d_model=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, spec_hidden, mol_hidden, spec_key_padding_mask=None, mol_key_padding_mask=None):
        # spec_key_padding_mask is unused: padded spectral positions are masked later by
        # the decoder's memory mask.
        mol_pad = None if mol_key_padding_mask is None else ~mol_key_padding_mask
        # need_weights keeps its default (True), which selects the math attention path.
        attended, _ = self.cross_attn(spec_hidden, mol_hidden, mol_hidden, key_padding_mask=mol_pad)
        return self.norm(spec_hidden + attended)

    @classmethod
    def from_config(cls, model_cfg: dict):
        cfg = {**SPECTRAL_DEFAULTS, **model_cfg}
        return cls(cfg["spec_d_model"], cfg["spec_num_heads"], cfg["spec_dropout"])


class PreRetrievalModel(nn.Module):
    """Stage 1: contrastive alignment of a trained spectral and a frozen Chemformer encoder."""

    def __init__(self, mol_encoder, spec_encoder, temperature=0.1):
        super().__init__()
        self.mol_encoder = mol_encoder
        self.spec_encoder = spec_encoder
        self.temperature = temperature
        for p in self.mol_encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode_molecules(self, token_ids, attention_mask=None):
        self.mol_encoder.eval()
        return F.normalize(self.mol_encoder(token_ids, attention_mask), dim=-1)

    def encode_spectra(self, spectra, mask=None):
        return F.normalize(self.spec_encoder(spectra, mask), dim=-1)

    def contrastive_loss(self, em, es, es_neg):
        """Mean of mol->spectrum (one intensity-noised negative) and CLIP spectrum->mol."""
        B, tau = em.shape[0], self.temperature
        logits_m2s = torch.stack([(em * es).sum(-1) / tau, (em * es_neg).sum(-1) / tau], dim=-1)
        loss_m2s = F.cross_entropy(logits_m2s, torch.zeros(B, dtype=torch.long, device=em.device))
        loss_s2m = F.cross_entropy(es @ em.T / tau, torch.arange(B, device=em.device))
        return 0.5 * (loss_m2s + loss_s2m)

    def forward(self, batch):
        em = self.encode_molecules(batch["smiles_ids"], batch["smiles_mask"])
        es = self.encode_spectra(batch["spectra"], batch["spec_mask"])
        es_neg = self.encode_spectra(batch["aug_spectra"], batch["spec_mask"])
        return self.contrastive_loss(em, es, es_neg)


class GenerativeRetrievalModel(nn.Module):
    """Stage 2: frozen encoders, trained cross-fusion and Chemformer decoder."""

    def __init__(self, mol_encoder, spec_encoder, cross_fusion, decoder, pad_token_idx=PAD_IDX):
        super().__init__()
        self.mol_encoder = mol_encoder
        self.spec_encoder = spec_encoder
        self.cross_fusion = cross_fusion
        self.decoder = decoder
        self.pad_token_idx = pad_token_idx
        for p in self.mol_encoder.parameters():
            p.requires_grad = False
        for p in self.spec_encoder.parameters():
            p.requires_grad = False

    def _encode_candidates(self, cand_ids, cand_mask):
        B, K, S = cand_ids.shape
        flat_ids = cand_ids.view(B * K, S)
        flat_mask = cand_mask.view(B * K, S)
        with torch.no_grad():
            self.mol_encoder.eval()
            hidden = self.mol_encoder.forward_sequence(flat_ids, flat_mask)
        return hidden.reshape(B, K * S, -1), flat_mask.reshape(B, K * S)

    def _fuse(self, spectra, spec_mask, cand_ids, cand_mask, hs):
        mol_hidden, mol_pad = self._encode_candidates(cand_ids, cand_mask)
        return self.cross_fusion(
            hs, mol_hidden, spec_key_padding_mask=spec_mask, mol_key_padding_mask=mol_pad
        )

    def train_step(self, batch):
        """Teacher-forced cross-entropy over the target SMILES tokens."""
        with torch.no_grad():
            self.spec_encoder.eval()
            hs = self.spec_encoder.forward_sequence(batch["spectra"], batch["spec_mask"])
        hca = self._fuse(batch["spectra"], batch["spec_mask"], batch["cand_ids"],
                         batch["cand_mask"], hs)
        tgt_ids, tgt_mask = batch["tgt_ids"], batch["tgt_mask"]
        logits = self.decoder(
            tgt_ids[:, :-1],
            hca,
            tgt_key_padding_mask=tgt_mask[:, :-1],
            memory_key_padding_mask=batch["spec_mask"],
        )
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt_ids[:, 1:].reshape(-1),
            ignore_index=self.pad_token_idx,
        )

    @torch.no_grad()
    def generate(self, spectra, spec_mask, cand_ids, cand_mask, max_len=512, beam_size=5):
        self.eval()
        hs = self.spec_encoder.forward_sequence(spectra, spec_mask)
        hca = self._fuse(spectra, spec_mask, cand_ids, cand_mask, hs)
        return self.decoder.generate(hca, spec_mask, max_len, beam_size)

    @torch.no_grad()
    def encode_molecules_cls(self, token_ids, attention_mask):
        self.mol_encoder.eval()
        return F.normalize(self.mol_encoder(token_ids, attention_mask), dim=-1)

    @torch.no_grad()
    def encode_spectra_pooled(self, spectra, mask):
        self.spec_encoder.eval()
        return F.normalize(self.spec_encoder(spectra, mask), dim=-1)


def load_torch(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def build_generative_model(chemformer_path, pre_retrieval, generative=None, model_cfg=None):
    """Assemble the two-stage model in GLMR's construction order.

    ``pre_retrieval`` and ``generative`` are checkpoint paths or loaded payloads; when
    ``generative`` is given its cross-fusion and decoder weights are loaded.
    """
    hp, state = load_chemformer_checkpoint(chemformer_path)
    pre = load_torch(pre_retrieval) if not isinstance(pre_retrieval, dict) else pre_retrieval
    gen = None
    if generative is not None:
        gen = load_torch(generative) if not isinstance(generative, dict) else generative
    if model_cfg is None:
        model_cfg = (gen or pre).get("config", {}).get("model", {})
    mol_encoder = ChemformerEncoder.from_chemformer(hp, state)
    mol_encoder.eval()
    spec_encoder = SpectralEncoder.from_config(model_cfg)
    spec_encoder.load_state_dict(pre["spec_encoder_state_dict"])
    spec_encoder.eval()
    cross_fusion = CrossFusion.from_config(model_cfg)
    decoder = ChemformerDecoder.from_chemformer(hp, state, encoder=mol_encoder)
    model = GenerativeRetrievalModel(
        mol_encoder, spec_encoder, cross_fusion, decoder,
        pad_token_idx=model_cfg.get("pad_token_idx", PAD_IDX),
    )
    if gen is not None:
        model.cross_fusion.load_state_dict(gen["cross_fusion_state_dict"])
        model.decoder.load_state_dict(gen["decoder_state_dict"])
    return model
