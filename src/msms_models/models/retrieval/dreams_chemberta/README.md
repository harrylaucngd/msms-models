# DreaMS + ChemBERTa alignment (SpecBridge)

Port of [HassounLab/SpecBridge](https://github.com/HassounLab/SpecBridge) (base revision
`6c6fadf`, plus the authors' local evaluation changes) as used for paper Table 3, row (1).
A spectrum is embedded by DreaMS: the masked mean of its peak embeddings is projected to
2048 dimensions, layer-normalised and L2-normalised. A Procrustes-initialised residual
mapper (`mapB`) then maps it into the CLS space of a frozen ChemBERTa. Candidates are
ranked by the dot product of the unnormalised mapped query `mu` with the unnormalised
ChemBERTa CLS embedding of each candidate SMILES.

| File | Contents |
|---|---|
| [`model.py`](model.py) | DreaMS backbone (forward only), adapter, mapper, ChemBERTa loader, strict checkpoint loading, `convert_checkpoint` |
| [`inference.py`](inference.py) | `predict`: pool cleaning, candidate and query embedding, ranking |
| [`candidates.py`](candidates.py) | Freezes the upstream evaluation's shuffled candidate order into an ID-keyed file |
| [`train.py`](train.py) | Upstream `train_real` for the settings of both paper checkpoints |
| [`vendor/`](vendor/) | DreaMS transformer, Fourier-feature and feed-forward layers (MIT) |

## Behaviour kept from upstream

- **Query batches.** DreaMS masks padded *query* rows only, so zero-padded peaks still act
  as attention keys. Embeddings therefore depend on how spectra are batched: queries are
  embedded in input order, `batch_size` (32) at a time, and padded to the batch maximum,
  as in the original evaluation. With `--limit`, the result matches the full run only
  when the limit is a multiple of 32.
- **Raw peaks.** The peaks enter DreaMS unchanged: no precursor token, no top-k peak
  filter, m/z divided by 1000. This reproduces the private `__normalize_spec` patch.
- **Candidates.** Upstream canonicalisation is a no-op (`_canon_smi` returns its input).
  Empty strings are dropped and duplicates are removed, keeping the first. Candidate
  embeddings are computed for the sorted unique strings in chunks of
  `embedding_batch_size` (512), with padding and truncation at 512 tokens.
- **Ranking.** Scores are ranked with `torch.argsort(scores, descending=True)`. For
  exactly tied scores, the order is the backend's. The official pools list the true
  molecule first. Upstream shuffled every pool with `random.seed(1234)`, sequentially over
  all pool keys in file order. [`candidates.py`](candidates.py) writes that order to the
  candidate file, so inference never needs the target.
- **No target access.** Upstream's forward pass also embeds the true SMILES, but that
  embedding is never used for ranking. The port does not compute it.

## Assets

| Configuration key | Input |
|---|---|
| `checkpoint` | A SpecBridge `last.pt`, or its inference-only copy from `model.convert_checkpoint`, which drops the optimizer state. Every weight is loaded strictly from it, ChemBERTa included. |
| `candidates` | An ID-keyed candidate file. For the paper rows, use the file frozen by `candidates.py`. |

**Parameters.** `chemberta_model` and `chemberta_revision` select the ChemBERTa tokenizer and
config. The model defaults to `Derify/ChemBERTa_augmented_pubchem_13m`; the revision has no
code default and is set by the configurations (`2da31509…`). No ChemBERTa weights are downloaded. `batch_size` defaults to 32 and
`embedding_batch_size` to 512.

## Run

```bash
python -m pip install -e '.[dreams-chemberta]'
msms-models predict --config paper/table3/dreams_chemberta/config-corrected.yaml --limit 32 --output results/smoke/dreams-chemberta
bash paper/table3/dreams_chemberta/run.sh
```

A GPU is recommended. When the unique candidates of the full test pool are embedded,
their CLS vectors take about 1.8 GB of host memory. A DreaMS batch whose largest spectrum
has 299 peaks builds a pairwise feature tensor of about 11 GB.

Training (the defaults are the stored `args` of both checkpoints):

```bash
python -m msms_models.models.retrieval.dreams_chemberta.train --config paper/table3/dreams_chemberta/train-corrected.yaml
python -m msms_models.models.retrieval.dreams_chemberta.train --config ... --limit 64 --max-steps 4   # smoke
```

`train.py` keeps upstream's construction order (DreaMS, adapter, ChemBERTa, projections,
mapper), so the global RNG initialises the trainable modules exactly as upstream does.
It also keeps upstream's module modes:

- DreaMS stays in the eval mode it is loaded in.
- A ChemBERTa built from its config (`chemberta_from_config`, the inflated run) stays in
  train mode, with dropout active.
- A ChemBERTa from `from_pretrained` is in eval mode.

The retained loss is `5 * MSE(mu, z_m) + 1e-3 * 1e-3 * orthogonality`. The InfoNCE and
SupCon terms have weight 0 or are commented out upstream. The data loader shuffles with
its own generator, seeded with the same 1234. The port drops wandb, the synthetic demo,
early stopping, AMP, resume and the binned-spectrum, fingerprint and formula features,
which the model never reads. Unlike upstream, it also saves `last.pt` when `max_steps`
stops training.

## Source and licenses

- **SpecBridge.** Adapted from `specbridge/adapters/dreams_adapter.py`,
  `models/mapper.py`, `losses/contrastive.py`, `eval/candidates.py`, `train/train.py`
  and the authors' `benchmark_msgym/compute_bootstrap_mces_msgym_retrained2.py`.
  The repository has an MIT badge but no LICENSE file.
- **DreaMS.** [pluskal-lab/DreaMS](https://github.com/pluskal-lab/DreaMS) at
  `02be141a2dd101e132a835ba6fea414048564a28`, which is identical to the copy vendored
  in SpecBridge; MIT, see [vendor/LICENSE.txt](vendor/LICENSE.txt). The package itself is
  not a dependency, for two reasons. Its pins conflict with Transformers 5:
  `huggingface_hub<1.0` against `>=1.3`, and `torch==2.2.1`. And its model module imports
  pandas, matchms, numba, plotly and igraph. The backbone in `model.py` keeps DreaMS's
  module names, so its checkpoints load strictly.
- **ChemBERTa.** [Derify/ChemBERTa_augmented_pubchem_13m](https://huggingface.co/Derify/ChemBERTa_augmented_pubchem_13m).
  Check the model card for its license.
