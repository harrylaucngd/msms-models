"""GLMR spectrum preparation, MGF training data, datasets and batching.

Ported from GLMR ``src/data/mgf_dataset.py``, ``src/data/gen_dataset.py`` and the
batching helpers of ``src/evaluate/eval_full.py``. Spectra are float32 arrays in
file order (MassSpecGym peaks are sorted by m/z); NumPy's default quicksort is used
everywhere as in the original, because it decides which equal-intensity peak survives
the top-``max_peaks`` cut.
"""

from __future__ import annotations

import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from .tokenizer import SmilesTokenizer


def parse_mgf(path) -> list[dict]:
    """GLMR's MGF reader: header fields as strings, peaks as float32 ``mzs``/``intensities``."""
    records, current, mzs, intensities, in_peaks = [], {}, [], [], False
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line == "BEGIN IONS":
                current, mzs, intensities, in_peaks = {}, [], [], False
            elif line == "END IONS":
                current["mzs"] = np.array(mzs, dtype=np.float32)
                current["intensities"] = np.array(intensities, dtype=np.float32)
                records.append(current)
            elif "=" in line and not in_peaks:
                key, value = line.split("=", 1)
                current[key.strip()] = value.strip()
            elif line and not line.startswith("#"):
                parts = line.split()
                if len(parts) >= 2:
                    in_peaks = True
                    mzs.append(float(parts[0]))
                    intensities.append(float(parts[1]))
    return records


def split_records(records):
    """Split by the ``FOLD`` field; a missing fold counts as training data."""
    folds = {"train": [], "val": [], "test": []}
    for record in records:
        fold = record.get("FOLD", "train")
        if fold in folds:
            folds[fold].append(record)
    return folds["train"], folds["val"], folds["test"]


def query_record(query: dict) -> dict:
    """A library query as GLMR's record: float32 peaks in their supplied (m/z) order."""
    spectrum = query.get("spectrum") or {}
    return {
        "mzs": np.asarray(spectrum.get("mz", []), dtype=np.float64).astype(np.float32),
        "intensities": np.asarray(spectrum.get("intensity", []), dtype=np.float64).astype(
            np.float32
        ),
    }


def prepare_spectrum(mzs, intensities, max_peaks=61) -> np.ndarray:
    """Keep the ``max_peaks`` most intense peaks, then sort by m/z; ``(T, 2)`` float32."""
    mzs, intensities = np.array(mzs, copy=True), np.array(intensities, copy=True)
    if len(mzs) > max_peaks:
        keep = np.argsort(intensities)[-max_peaks:]
        mzs, intensities = mzs[keep], intensities[keep]
    order = np.argsort(mzs)
    return np.stack([mzs[order], intensities[order]], axis=-1).astype(np.float32)


def load_pickle(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


class ContrastiveDataset(Dataset):
    """Stage-1 items: spectrum, its intensity-noised copy (the negative) and the SMILES."""

    def __init__(
        self,
        records,
        tokenizer: SmilesTokenizer,
        max_peaks=61,
        max_smiles_len=512,
        augment_intensity=True,
        noise_scale=0.3,
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.max_peaks = max_peaks
        self.max_smiles_len = max_smiles_len
        self.augment_intensity = augment_intensity
        self.noise_scale = noise_scale

    def __len__(self):
        return len(self.records)

    def _augment(self, spec):
        # Global NumPy RNG, seeded per DataLoader worker by PyTorch, as in GLMR.
        augmented = spec.copy()
        noise = 1.0 + np.random.uniform(-self.noise_scale, self.noise_scale, size=spec.shape[0])
        augmented[:, 1] = np.clip(augmented[:, 1] * noise, 0.0, 1.0)
        return augmented

    def __getitem__(self, index):
        record = self.records[index]
        spec = prepare_spectrum(record["mzs"], record["intensities"], self.max_peaks)
        smiles_ids = self.tokenizer.encode(record["SMILES"], max_len=self.max_smiles_len)
        augmented = self._augment(spec) if self.augment_intensity else spec.copy()
        return {
            "spectrum": torch.from_numpy(spec),
            "aug_spectrum": torch.from_numpy(augmented),
            "smiles_ids": torch.tensor(smiles_ids, dtype=torch.long),
            "num_peaks": spec.shape[0],
        }


def collate_contrastive(batch):
    size = len(batch)
    peaks = max(b["num_peaks"] for b in batch)
    length = max(b["smiles_ids"].shape[0] for b in batch)
    out = {
        "spectra": torch.zeros(size, peaks, 2),
        "aug_spectra": torch.zeros(size, peaks, 2),
        "spec_mask": torch.zeros(size, peaks, dtype=torch.bool),
        "smiles_ids": torch.zeros(size, length, dtype=torch.long),
        "smiles_mask": torch.zeros(size, length, dtype=torch.bool),
    }
    for i, b in enumerate(batch):
        n, s = b["num_peaks"], b["smiles_ids"].shape[0]
        out["spectra"][i, :n] = b["spectrum"]
        out["aug_spectra"][i, :n] = b["aug_spectrum"]
        out["spec_mask"][i, :n] = True
        out["smiles_ids"][i, :s] = b["smiles_ids"]
        out["smiles_mask"][i, :s] = True
    return out


class GenerativeDataset(Dataset):
    """Stage-2 items: spectrum, its precomputed top-K candidates and the target SMILES.

    ``precomputed`` maps the record's index within its split to the candidate list.
    """

    def __init__(self, records, precomputed, tokenizer, max_peaks=61, max_smiles_len=512,
                 top_k=40):
        self.tokenizer = tokenizer
        self.max_peaks = max_peaks
        self.max_smiles_len = max_smiles_len
        self.top_k = top_k
        self.samples = [
            (record, precomputed[index])
            for index, record in enumerate(records)
            if index in precomputed
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        record, candidates = self.samples[index]
        spec = prepare_spectrum(record["mzs"], record["intensities"], self.max_peaks)
        candidates = candidates[: self.top_k]
        while len(candidates) < self.top_k:
            candidates.append(candidates[-1] if candidates else record["SMILES"])
        return {
            "spectrum": torch.from_numpy(spec),
            "num_peaks": spec.shape[0],
            "cand_ids_list": [
                self.tokenizer.encode(s, max_len=self.max_smiles_len) for s in candidates
            ],
            "tgt_ids": torch.tensor(
                self.tokenizer.encode(record["SMILES"], max_len=self.max_smiles_len),
                dtype=torch.long,
            ),
        }


def collate_generative(batch):
    size = len(batch)
    peaks = max(b["num_peaks"] for b in batch)
    target = max(b["tgt_ids"].shape[0] for b in batch)
    k = len(batch[0]["cand_ids_list"])
    length = max([1, *(len(c) for b in batch for c in b["cand_ids_list"])])
    out = {
        "spectra": torch.zeros(size, peaks, 2),
        "spec_mask": torch.zeros(size, peaks, dtype=torch.bool),
        "cand_ids": torch.zeros(size, k, length, dtype=torch.long),
        "cand_mask": torch.zeros(size, k, length, dtype=torch.bool),
        "tgt_ids": torch.zeros(size, target, dtype=torch.long),
        "tgt_mask": torch.zeros(size, target, dtype=torch.bool),
    }
    for i, b in enumerate(batch):
        n = b["num_peaks"]
        out["spectra"][i, :n] = b["spectrum"]
        out["spec_mask"][i, :n] = True
        for j, ids in enumerate(b["cand_ids_list"]):
            out["cand_ids"][i, j, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            out["cand_mask"][i, j, : len(ids)] = True
        t = b["tgt_ids"].shape[0]
        out["tgt_ids"][i, :t] = b["tgt_ids"]
        out["tgt_mask"][i, :t] = True
    return out


def batch_spectra(spectra, device):
    """Pad ``(T_i, 2)`` arrays to ``(B, T_max, 2)`` with a True-for-peaks mask."""
    length = max(s.shape[0] for s in spectra)
    padded = torch.zeros(len(spectra), length, 2, device=device)
    mask = torch.zeros(len(spectra), length, dtype=torch.bool, device=device)
    for i, s in enumerate(spectra):
        padded[i, : s.shape[0]] = torch.from_numpy(s)
        mask[i, : s.shape[0]] = True
    return padded, mask


def batch_tokens(token_lists, device=None):
    """Right-pad token id lists to ``(N, S_max)`` ids and a True-for-tokens mask."""
    length = max(len(t) for t in token_lists)
    ids = torch.zeros(len(token_lists), length, dtype=torch.long, device=device)
    mask = torch.zeros(len(token_lists), length, dtype=torch.bool, device=device)
    for j, tokens in enumerate(token_lists):
        ids[j, : len(tokens)] = torch.tensor(tokens)
        mask[j, : len(tokens)] = True
    return ids, mask


def batch_candidates(candidate_lists, tokenizer, device, k, max_len=512):
    """Tokenize K candidates per sample to ``(B, K, S_max)``, repeating the last to fill K."""
    token_lists, length = [], 1
    for smiles_list in candidate_lists:
        sample = [tokenizer.encode(s, max_len=max_len) for s in smiles_list[:k]]
        length = max([length, *(len(t) for t in sample)])
        while len(sample) < k:
            sample.append(sample[-1])
        token_lists.append(sample)
    ids = torch.zeros(len(candidate_lists), k, length, dtype=torch.long, device=device)
    mask = torch.zeros(len(candidate_lists), k, length, dtype=torch.bool, device=device)
    for i, sample in enumerate(token_lists):
        for j, tokens in enumerate(sample):
            ids[i, j, : len(tokens)] = torch.tensor(tokens)
            mask[i, j, : len(tokens)] = True
    return ids, mask
