# Table 5 · SMILES Transformer, SELFIES Transformer

| | Acc@1 | MCES@1 | Tanimoto@1 | Acc@10 | MCES@10 | Tanimoto@10 |
|---|---|---|---|---|---|---|
| SMILES Transformer [8] | 0.00% | 79.39 | 0.03 | 0.00% | 52.13 | 0.10 |
| SELFIES Transformer [8] | 0.00% | 38.88 | 0.08 | 0.00% | 26.87 | 0.13 |

**Source.** [`models/de_novo/smiles_transformer`](../../../src/msms_models/models/de_novo/smiles_transformer/) and [`massspecgym/`](../../../src/msms_models/massspecgym/) ([MassSpecGym](https://github.com/pluskal-lab/MassSpecGym)).

**What runs.** An encoder–decoder transformer reads the spectrum's peaks (the precursor first, peaks from m/z 10 to 1005, the 60 most intense) with the query's formula added to every peak, and writes a molecule token by token. The SMILES row uses a byte-level BPE vocabulary learnt from 4M MassSpecGym molecules and the training molecules (4 heads, 3 encoder layers); the SELFIES row uses the SELFIES alphabet of the training molecules (8 heads, 6 encoder layers). Ten molecules per spectrum are ten independent samples at temperature 1. `run.sh` first builds the tokenizer and trains the model: cross entropy with teacher forcing, Adam at 3·10⁻⁴, a global batch of 512, one validation per epoch, early stopping after three without improvement, and the checkpoint with the lowest validation loss.

**Inputs.** Official MassSpecGym training, validation and test queries with known formulas (`bash paper/prepare_data.sh`); for the SMILES row, the 4M molecule file, which `run.sh` downloads.

## Run

From the repository root:

```bash
DEVICE=cuda:0 bash paper/table5/smiles_transformer/run.sh
```

Set `OUT=...` to choose the output directory.
`run.sh` predicts and evaluates with the fixed `table5` protocol.
