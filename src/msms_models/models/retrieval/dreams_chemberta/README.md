# DreaMS + ChemBERTa alignment (SpecBridge)

SpecBridge as used for paper Table 3, row (1). A spectrum is embedded by DreaMS: the masked
mean of its peak embeddings is projected to 2048 dimensions, layer-normalised and
L2-normalised. A Procrustes-initialised residual mapper (`mapB`) then maps it into the CLS
space of a frozen ChemBERTa. Candidates are ranked by the dot product of the unnormalised
mapped query `mu` with the unnormalised ChemBERTa CLS embedding of each candidate SMILES.

| File | Contents |
|---|---|
| [`model.py`](model.py) | DreaMS backbone (forward only), adapter, mapper, ChemBERTa loader, strict checkpoint loading, `convert_checkpoint` |
| [`inference.py`](inference.py) | `predict`: pool cleaning, candidate and query embedding, ranking |
| [`candidates.py`](candidates.py) | Writes each official pool in SpecBridge's shuffled evaluation order to an ID-keyed file |
| [`train.py`](train.py) | Training with the settings of both paper checkpoints |
| [`vendor/`](vendor/) | DreaMS transformer, Fourier-feature and feed-forward layers (MIT) |

- **Queries.** Peaks enter DreaMS unchanged: no precursor token, no top-k peak filter, m/z
  divided by 1000. DreaMS masks padded query rows only, so zero-padded peaks still act as
  attention keys and embeddings depend on the batching: queries are embedded in input
  order, `batch_size` (32) at a time, padded to the batch maximum.
- **Candidates.** SMILES are used as given. Empty strings are dropped and duplicates are
  removed, keeping the first. Embeddings are computed for the sorted unique strings in
  chunks of `embedding_batch_size` (512), with padding and truncation at 512 tokens.
- **Ranking.** Scores are ranked with `torch.argsort(scores, descending=True)`. The
  official pools list the true molecule first; [`candidates.py`](candidates.py) shuffles
  every pool with `random.seed(1234)`, sequentially over all pool keys in file order, and
  inference reads that order. The target is never read at inference.

## Assets

| Configuration key | Input |
|---|---|
| `checkpoint` | A SpecBridge `last.pt`, or its inference-only copy from `model.convert_checkpoint`, which drops the optimizer state. Every weight is loaded strictly from it, ChemBERTa included. |
| `candidates` | An ID-keyed candidate file written by `candidates.py`. |

**Parameters.** `chemberta_model` and `chemberta_revision` select the ChemBERTa tokenizer and
config. The model defaults to `Derify/ChemBERTa_augmented_pubchem_13m`; the revision has no
code default and is set by the configurations (`2da31509…`). No ChemBERTa weights are
downloaded. `batch_size` defaults to 32 and `embedding_batch_size` to 512.

## Run

From the repository root, after `bash paper/prepare_data.sh`:

```bash
bash paper/table3/dreams_chemberta/run.sh
```

`run.sh` writes the shuffled candidate files, then evaluates the inflated and corrected rows.
A GPU is recommended. When the unique candidates of the full test pool are embedded,
their CLS vectors take about 1.8 GB of host memory. A DreaMS batch whose largest spectrum
has 299 peaks builds a pairwise feature tensor of about 11 GB.

## Training

The defaults are the stored `args` of both checkpoints:

```bash
python -m msms_models.models.retrieval.dreams_chemberta.train --config paper/table3/dreams_chemberta/train-corrected.yaml
```

`paper/table3/dreams_chemberta/train-inflated.yaml` trains the inflated row. Module modes
during training:

- DreaMS stays in eval mode.
- A ChemBERTa built from its config (`chemberta_from_config`, the inflated run) is in
  train mode, with dropout active.
- A ChemBERTa from `from_pretrained` is in eval mode.

The loss is `5 * MSE(mu, z_m) + 1e-3 * 1e-3 * orthogonality`. The data loader shuffles with
its own generator, seeded with 1234. `last.pt` is saved at the end of training, also when
`max_steps` stops it.

## Source

- **SpecBridge.** [HassounLab/SpecBridge](https://github.com/HassounLab/SpecBridge) at
  `6c6fadf`, with the authors' evaluation changes: `specbridge/adapters/dreams_adapter.py`,
  `models/mapper.py`, `losses/contrastive.py`, `eval/candidates.py`, `train/train.py` and
  `benchmark_msgym/compute_bootstrap_mces_msgym_retrained2.py`. The repository has an MIT
  badge but no LICENSE file.
- **DreaMS.** [pluskal-lab/DreaMS](https://github.com/pluskal-lab/DreaMS) at
  `02be141a2dd101e132a835ba6fea414048564a28`, MIT; see [vendor/LICENSE.txt](vendor/LICENSE.txt).
  The backbone in `model.py` keeps DreaMS's module names, so its checkpoints load strictly.
- **ChemBERTa.** [Derify/ChemBERTa_augmented_pubchem_13m](https://huggingface.co/Derify/ChemBERTa_augmented_pubchem_13m).
  Check the model card for its license.

See also the [third-party notices](../../../../../THIRD_PARTY_NOTICES.md).
