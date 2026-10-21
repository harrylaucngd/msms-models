"""SpecBridge modules: a DreaMS spectrum adapter mapped into a frozen ChemBERTa space.

Adapted from HassounLab/SpecBridge (`specbridge/adapters/dreams_adapter.py`,
`specbridge/models/mapper.py`, `specbridge/losses/contrastive.py`). Module and
attribute names follow the original so that its checkpoints load strictly.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from .vendor.feed_forward import FeedForward
from .vendor.fourier_features import FourierFeatures
from .vendor.layers import TransformerEncoder

# DreaMS backbone of `ssl_model.ckpt` (DataFormatA: max m/z 1000, TBXIC bound 1e-4).
DREAMS_MAX_MZ = 1000.0
DREAMS_ARGS = SimpleNamespace(
    n_layers=7,
    n_heads=8,
    d_model=1024,  # d_fourier 980 + d_peak 44
    att_dropout=0.1,
    ff_dropout=0.1,
    residual_dropout=0.1,
    no_transformer_bias=True,
    attn_mech="dot-product",
    d_graphormer_params=0,  # graphormer_mz_diffs=True, graphormer_parametrized=False
    pre_norm=True,
    scnorm=False,
)
UNUSED_DREAMS_HEADS = ("ff_out.", "mz_masking_loss.", "ro_out.")


def unit_normalize(x, eps=1e-8):
    return x / (x.norm(dim=-1, keepdim=True) + eps)


class DreaMSEncoder(nn.Module):
    """The DreaMS backbone (`dreams.models.dreams.dreams.DreaMS`) without its training heads."""

    def __init__(self, *, build_pretraining_head=False):
        super().__init__()
        self.fourier_enc = FourierFeatures(
            strategy="lin_float_int", x_min=1e-4, x_max=DREAMS_MAX_MZ, trainable=False,
            num_freqs=None,
        )
        self.ff_fourier = FeedForward(
            in_dim=self.fourier_enc.num_features(), out_dim=980, dropout=0.1, depth=5,
            hidden_dim=512, bias=True,
        )
        self.ff_peak = FeedForward(in_dim=2, hidden_dim=44, out_dim=44, depth=1, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(DREAMS_ARGS)
        self.embed_dim = DREAMS_ARGS.d_model
        if build_pretraining_head:
            # `ssl_model.ckpt` (train_objective mask_mz_hot, ret_order_loss_w 0.2) also
            # creates a masked-m/z head and a retention-order head, which SpecBridge
            # deletes (`PreTrainedModel.remove_unused_backbone_parameters`). Creating and
            # discarding them keeps the global torch RNG stream, and hence the
            # initialisation of the trainable modules built afterwards, as in training.
            FeedForward(
                in_dim=1024, hidden_dim=1024, depth=1, act_last=False, out_dim=20000,
                dropout=0.1, bias=True,
            )
            nn.Linear(2 * DREAMS_ARGS.d_model, 1, bias=False)

    def forward(self, spec):
        """Peak embeddings of padded `[batch, peaks, (m/z, intensity)]` spectra."""
        # As in DreaMS, padding is m/z == 0. DreaMS masks padded *query* rows only,
        # so padded tokens still act as keys: embeddings depend on the batch padding.
        padding_mask = spec[:, :, 0] == 0
        # SpecBridge replaces DreaMS's private `__normalize_spec` with this version,
        # which takes device and dtype from the input instead of the LightningModule.
        scale = torch.tensor([DREAMS_MAX_MZ, 1.0], device=spec.device, dtype=spec.dtype)
        peak_embs = self.ff_peak(spec / scale)
        fourier_features = self.ff_fourier(self.fourier_enc(spec[..., [0]]))
        x = torch.cat([peak_embs, fourier_features], dim=-1)
        graphormer_dists = fourier_features.unsqueeze(2) - fourier_features.unsqueeze(1)
        return self.transformer_encoder(x, padding_mask, graphormer_dists)


def _load_lightning_checkpoint(path):
    """Read a Lightning checkpoint without importing the classes in its hyperparameters."""
    import pickle

    class Placeholder:
        def __init__(self, *args, **kwargs):
            pass

        def __setstate__(self, state):
            pass

    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module.split(".")[0] in {"dreams", "pytorch_lightning", "lightning_fabric"}:
                return Placeholder
            return super().find_class(module, name)

    module = SimpleNamespace(Unpickler=Unpickler, load=pickle.load, __name__="pickle")
    return torch.load(path, map_location="cpu", weights_only=False, pickle_module=module)


def load_dreams_checkpoint(path, *, build_pretraining_head=True):
    """DreaMS SSL weights from a Lightning checkpoint (e.g. Zenodo 10997887 `ssl_model.ckpt`)."""
    payload = _load_lightning_checkpoint(path)
    state = payload.get("state_dict", payload)
    state = {k: v for k, v in state.items() if not k.startswith(UNUSED_DREAMS_HEADS)}
    model = DreaMSEncoder(build_pretraining_head=build_pretraining_head)
    model.load_state_dict(state, strict=True)
    return model.eval()  # PreTrainedModel.from_ckpt returns the backbone in eval mode


class DreamsAdapter(nn.Module):
    """Masked mean of DreaMS peak embeddings, projected, layer-normalised and L2-normalised."""

    def __init__(self, dreams_encoder, d_out, hidden=0):
        super().__init__()
        self.dreams = dreams_encoder
        d_in = dreams_encoder.embed_dim
        self.proj = nn.Sequential(
            nn.Linear(d_in, d_out) if hidden == 0 else nn.Sequential(
                nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, d_out)
            ),
            nn.LayerNorm(d_out),
        )
        for p in self.dreams.parameters():
            p.requires_grad = False

    def forward(self, peaks):
        if any(p.requires_grad for p in self.dreams.parameters()):
            out = self.dreams(peaks)
        else:
            with torch.no_grad():
                out = self.dreams(peaks)
        mask = (peaks[..., 0] > 0).float()
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (out * mask.unsqueeze(-1)).sum(dim=1) / denom
        return unit_normalize(self.proj(pooled))

    def unfreeze_last(self, n_layers):
        """Make the last attention and feed-forward blocks of DreaMS trainable."""
        encoder = self.dreams.transformer_encoder
        count = min(n_layers, len(encoder.atts))
        for layer in [*encoder.atts[-count:], *encoder.ffs[-count:]]:
            for p in layer.parameters():
                p.requires_grad = True


class ResidualBlock(nn.Module):
    def __init__(self, d, hidden, drop=0.0):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, hidden)
        self.fc2 = nn.Linear(hidden, d)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()
        nn.init.xavier_uniform_(self.fc1.weight, gain=0.7)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        return x + self.fc2(self.drop(self.act(self.fc1(self.ln(x)))))


class ProcrustesResidualMapper(nn.Module):
    """Spectrum to molecule space: an orthogonally initialised linear map plus residual MLPs."""

    def __init__(self, d_in, d_out, n_blocks=4, hidden=0, drop=0.0):
        super().__init__()
        self.W = nn.Linear(d_in, d_out, bias=True)
        nn.init.orthogonal_(self.W.weight)
        nn.init.zeros_(self.W.bias)
        hidden = hidden if hidden > 0 else max(256, min(d_out, 1024))
        self.blocks = nn.ModuleList([ResidualBlock(d_out, hidden, drop) for _ in range(n_blocks)])

    def forward(self, x):
        mu = self.W(x)
        for block in self.blocks:
            mu = block(mu)
        return mu  # unnormalised; the deterministic (non-Gaussian) query embedding

    def orthogonality_penalty(self, strength=1e-3):
        W = self.W.weight
        G = W @ W.T if W.size(0) <= W.size(1) else W.T @ W
        eye = torch.eye(G.size(0), device=W.device, dtype=W.dtype)
        return strength * (G - eye).pow(2).mean()


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.log_tau = nn.Parameter(torch.log(torch.tensor(temperature)))

    def forward(self, z_s, z_m):
        logits = unit_normalize(z_s) @ unit_normalize(z_m).T
        logits = logits / torch.exp(self.log_tau).clamp(min=1e-6)
        loss_i = -torch.diagonal(F.log_softmax(logits, dim=1)).mean()
        loss_t = -torch.diagonal(F.log_softmax(logits.T, dim=1)).mean()
        return 0.5 * (loss_i + loss_t)


def select_backbone_state_dict(raw_state, expected_keys):
    """Map checkpoint keys with nested wrapper prefixes onto RoBERTa backbone keys."""
    strip_prefixes = ("module.", "model.", "_orig_mod.", "roberta.", "chem_mdl.")
    matched = {}
    for raw_key, value in raw_state.items():
        queue, seen, candidates = [raw_key], {raw_key}, []
        while queue:
            current = queue.pop()
            candidates.append(current)
            for prefix in strip_prefixes:
                if current.startswith(prefix) and current[len(prefix):] not in seen:
                    seen.add(current[len(prefix):])
                    queue.append(current[len(prefix):])
        for key in sorted(candidates, key=len, reverse=True):
            if key in expected_keys:
                matched[key] = value
                break
    return matched


def load_chemberta(name, *, revision=None, from_config=False, checkpoint=None):
    """Tokenizer and RoBERTa encoder, optionally overwritten by a continued-pretraining state."""
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    if from_config:
        # The inflated run's model directory held only config and tokenizer files, so
        # SpecBridge built the encoder from its config and loaded `checkpoint` into it.
        # Such a model stays in train mode (dropout active) because nothing calls eval().
        model = AutoModel.from_config(AutoConfig.from_pretrained(name, revision=revision))
        model.init_weights()
    else:
        model = AutoModel.from_pretrained(name, revision=revision)
    if checkpoint is not None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        raw = next(
            (payload[k] for k in ("model", "state_dict", "model_state_dict", "module")
             if isinstance(payload.get(k), dict)),
            None,
        )
        if raw is None and all(torch.is_tensor(v) for v in payload.values()):
            raw = payload
        model.load_state_dict(select_backbone_state_dict(raw, set(model.state_dict())), strict=False)
    for p in model.parameters():
        p.requires_grad = False
    return tokenizer, model


class SpecBridge(nn.Module):
    """SpecBridge `DreamsToMolCondition` with `mol_space=chemberta` and a non-Gaussian mapper."""

    def __init__(self, dreams_encoder, chemberta, *, cond_dim=2048, hidden=2048, n_blocks=8):
        super().__init__()
        self.spec = DreamsAdapter(dreams_encoder, d_out=cond_dim, hidden=hidden)
        # `chemberta` is called here, between the trainable modules, to keep the
        # original construction (and RNG) order.
        self.chem_tok, self.chem_mdl = chemberta()
        width = int(self.chem_mdl.config.hidden_size)
        # Trained but unused: no retained loss term reads this projection.
        self.chem_proj = nn.Sequential(
            nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, cond_dim)),
            nn.LayerNorm(cond_dim),
        )
        self.mapB = ProcrustesResidualMapper(cond_dim, width, n_blocks=n_blocks, hidden=hidden)
        self.contrast = InfoNCELoss(temperature=0.07)

    def train(self, mode=True):
        super().train(mode)
        if not any(p.requires_grad for p in self.chem_mdl.parameters()):
            self.chem_mdl.eval()
        return self

    def embed_smiles(self, smiles, device):
        """ChemBERTa CLS state of the last layer, unnormalised."""
        tokens = self.chem_tok(smiles, padding=True, truncation=True, return_tensors="pt")
        tokens = tokens.to(device)
        if any(p.requires_grad for p in self.chem_mdl.parameters()):
            return self.chem_mdl(**tokens).last_hidden_state[:, 0]
        with torch.no_grad():
            return self.chem_mdl(**tokens).last_hidden_state[:, 0]

    def embed_spectra(self, peaks):
        """Returns the adapter embedding `z_s` and the mapped query `mu` in ChemBERTa space."""
        z_s = self.spec(peaks)
        return z_s, self.mapB(z_s)

    def align_losses(self, mu, z_m, *, w_map=5.0, w_ortho=1e-3, w_con_mapped=0.0):
        """SpecBridge's retained objective: InfoNCE(mu, z_m) + orthogonality + w_map * MSE."""
        if w_con_mapped > 0:
            con = self.contrast(mu, z_m.detach())
        else:
            con = torch.tensor(0.0, device=mu.device)
        mse = F.mse_loss(mu, z_m, reduction="mean")
        # As upstream: the penalty's default 1e-3 strength is scaled again by w_ortho.
        ortho = self.mapB.orthogonality_penalty() * w_ortho
        loss = w_con_mapped * con + ortho + w_map * mse
        return loss, {"L_con_m": con.detach(), "L_ortho": ortho.detach(), "L_map": mse.detach()}


def pad_peaks(spectra, device=None):
    """`[batch, max peaks, 2]` float32 tensor padded with zeros, as SpecBridge's collate."""
    tensors = [
        torch.stack(
            [torch.tensor(s["mz"], dtype=torch.float32),
             torch.tensor(s["intensity"], dtype=torch.float32)],
            dim=-1,
        )
        for s in spectra
    ]
    peaks = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=0.0)
    return peaks if device is None else peaks.to(device)


def read_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def build_model(checkpoint, *, chemberta_model, chemberta_revision=None, device="cpu"):
    """Inference model with every weight, ChemBERTa included, read strictly from `checkpoint`."""
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    payload = read_checkpoint(checkpoint)
    args = payload.get("args", {})

    def chemberta():
        tokenizer = AutoTokenizer.from_pretrained(chemberta_model, revision=chemberta_revision)
        config = AutoConfig.from_pretrained(chemberta_model, revision=chemberta_revision)
        return tokenizer, AutoModel.from_config(config)

    model = SpecBridge(
        DreaMSEncoder(),
        chemberta,
        cond_dim=int(args.get("cond_dim", 2048)),
        hidden=int(args.get("mapper_hidden", 2048)),
        n_blocks=int(args.get("n_blocks", 8)),
    )
    model.load_state_dict(payload["model"], strict=True)
    for p in model.parameters():
        p.requires_grad = False
    return model.to(device).eval(), args


def convert_checkpoint(source, destination):
    """Write an inference-only copy of a SpecBridge `last.pt` (drops the optimizer state)."""
    payload = torch.load(source, map_location="cpu", weights_only=True)
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": payload["model"], "args": payload.get("args", {})}, destination)
