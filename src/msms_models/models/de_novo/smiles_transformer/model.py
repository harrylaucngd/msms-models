"""The SMILES Transformer network of MassSpecGym (``models/de_novo/smiles_tranformer.py``)."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ....massspecgym.layers import mlp
from ....massspecgym.transforms import ELEMENTS


class SmilesTransformer(nn.Module):
    """Spectrum peaks (m/z, intensity) to SMILES or SELFIES tokens with ``nn.Transformer``.

    There is no positional encoding; source and target embeddings are scaled by √d_model; the
    formula embedding (one linear layer of the 118 element counts) is added to every peak
    before that scaling; padded peaks are masked in the encoder but not in the decoder's
    cross-attention.
    """

    def __init__(self, vocab_size, pad_id, d_model=256, nhead=4, num_encoder_layers=3,
                 num_decoder_layers=4, dropout=0.1, input_dim=2, pre_norm=False):
        super().__init__()
        self.d_model, self.pad_id = d_model, pad_id
        self.src_encoder = nn.Linear(input_dim, d_model)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers, dim_feedforward=4 * d_model, dropout=dropout,
            norm_first=pre_norm,
        )
        self.tgt_decoder = nn.Linear(d_model, vocab_size)
        self.formula_mlp = mlp(len(ELEMENTS), len(ELEMENTS), d_model, 1, dropout)

    def encode(self, spec, formula):
        src = self.src_encoder(spec.permute(1, 0, 2))  # (peaks, batch, d_model)
        src = (src + self.formula_mlp(formula).unsqueeze(0)) * (self.d_model ** 0.5)
        return self.transformer.encoder(src, src_key_padding_mask=spec.sum(-1) == 0)

    def forward(self, spec, formula, tokens):
        """Teacher forcing: logits for ``tokens[:, 1:]`` from ``tokens[:, :-1]``, flattened."""
        tokens = tokens.permute(1, 0)  # (length, batch)
        tgt = self.tgt_embedding(tokens[:-1]) * (self.d_model ** 0.5)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tokens.size(0)).to(spec.device)
        output = self.transformer.decoder(
            tgt, self.encode(spec, formula), tgt_mask=tgt_mask[:-1, :-1],
            tgt_key_padding_mask=(tokens == self.pad_id).permute(1, 0)[:, :-1],
        )
        logits = self.tgt_decoder(output)
        return logits.view(-1, logits.size(-1)), tokens[1:].contiguous().view(-1)

    @torch.inference_mode()
    def decode(self, spec, formula, *, sos_id, eos_id, max_len, temperature):
        """``greedy_decode``: argmax if ``temperature`` is None, else sampling at ``temperature``.

        Stops early only when every sequence emits ``</s>`` at the same step.
        """
        memory = self.encode(spec, formula)
        tokens = torch.full((1, spec.size(0)), sos_id, dtype=torch.long, device=spec.device)
        for _ in range(max_len - 1):
            tgt = self.tgt_embedding(tokens) * (self.d_model ** 0.5)
            tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(0)).to(spec.device)
            logits = self.tgt_decoder(self.transformer.decoder(tgt, memory, tgt_mask=tgt_mask)[-1])
            if temperature is None:
                next_token = torch.argmax(F.softmax(logits, dim=-1), dim=-1)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=0)
            if torch.all(next_token == eos_id):
                break
        return tokens.permute(1, 0)
