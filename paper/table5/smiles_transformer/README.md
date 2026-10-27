# Table 5 · SMILES Transformer, SELFIES Transformer

| | Acc@1 | MCES@1 | Tanimoto@1 | Acc@10 | MCES@10 | Tanimoto@10 |
|---|---|---|---|---|---|---|
| SMILES Transformer [8] | 0.00% | 79.39 | 0.03 | 0.00% | 52.13 | 0.10 |
| SELFIES Transformer [8] | 0.00% | 38.88 | 0.08 | 0.00% | 26.87 | 0.13 |

**Source.** [`models/de_novo/smiles_transformer`](../../../src/msms_models/models/de_novo/smiles_transformer/) and [`massspecgym/`](../../../src/msms_models/massspecgym/), reimplementing [MassSpecGym](https://github.com/pluskal-lab/MassSpecGym)'s `SmilesTransformer` and tokenizers at `485deb22df`. The printed rows are MassSpecGym's.

**What runs.** An encoder–decoder transformer reads the spectrum's peaks (the precursor first, peaks from m/z 10 to 1005, the 60 most intense) with the query's formula added to every peak, and writes a molecule token by token. The SMILES row uses a byte-level BPE vocabulary learnt from 4M MassSpecGym molecules and the training molecules (4 heads, 3 encoder layers); the SELFIES row uses the SELFIES alphabet of the training molecules (8 heads, 6 encoder layers). Ten molecules per spectrum are ten independent samples at temperature 1. MassSpecGym released no checkpoints, so `run.sh` builds the tokenizer and trains the model with the settings of MassSpecGym's formula-conditioned ("v3") runs: cross entropy with teacher forcing, Adam at 3·10⁻⁴, a global batch of 512 (8 GPUs × 64), one validation per epoch, early stopping after three without improvement, and the checkpoint with the lowest validation loss. A retrained model reproduces the printed numbers within their uncertainty, not to the digit.

**Inputs.** Official MassSpecGym training, validation and test queries with known formulas (`bash paper/prepare_data.sh`); for the SMILES row, the 4M molecule file, which `run.sh` downloads.

**Environment.** The locked environment (`bash environments/install.sh`; see [installation](../../../docs/install.md)). A GPU is needed for training.

## Run

From the repository root:

```bash
bash paper/table5/smiles_transformer/run.sh
```

Set `DEVICE=cuda:0` for GPU training and inference and `OUT=...` to choose the output directory.
`run.sh` predicts and evaluates with the fixed `table5` protocol.
