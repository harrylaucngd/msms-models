# Table 3 · (1) DreaMS + ChemBERTa alignment (SpecBridge)

| | Hit@1 | Hit@5 | Hit@20 | MCES@1 |
|---|---|---|---|---|
| inflated (diagnostic) | 82.41% (81.46–83.33) | 92.57% (91.91–93.21) | 97.41% (97.00–97.78) | 2.86 (2.70–3.03) |
| corrected | 6.70% (6.09–7.35) | 19.82% (18.89–20.83) | 40.62% (39.44–41.79) | 13.13 (12.97–13.28) |

Intervals are 99.9% BCa over 20,000 spectrum-level resamples.

**Source.** [`models/retrieval/dreams_chemberta`](../../../src/msms_models/models/retrieval/dreams_chemberta/),
a port of [HassounLab/SpecBridge](https://github.com/HassounLab/SpecBridge) as run by the
authors. DreaMS layers come from [pluskal-lab/DreaMS](https://github.com/pluskal-lab/DreaMS).

**What runs.** DreaMS embeds each spectrum. The masked mean of its peak embeddings is
projected and L2-normalised, and a residual mapper places it in the CLS space of a frozen
ChemBERTa. Candidates are ranked by the dot product with their ChemBERTa CLS embeddings.
The two rows share code and settings and differ only in their inputs:

- **Inflated.** The raw MassSpecGym strings. The true SMILES keeps its repository (Kekulé)
  spelling while the decoys are RDKit-canonical. The alignment uses a ChemBERTa further
  MLM-trained on canonical SMILES, so the string style alone identifies the answer.
- **Corrected.** Every query and candidate string is RDKit-canonical (MassSpecGym 1.5),
  with the public ChemBERTa.

Both models were trained for 2 epochs (batch size 32, lr 1e-4, seed 1234) with the last two
DreaMS blocks unfrozen and ChemBERTa frozen; see [`train-inflated.yaml`](train-inflated.yaml)
and [`train-corrected.yaml`](train-corrected.yaml).

**Inputs.**
- **Queries:** the official test spectra.
- **Candidates:** the official original pool (inflated) and 1.5 pool (corrected). `run.sh` writes them in SpecBridge's evaluation order: `random.seed(1234)`, then one shuffle per pool in file order.
- **Weights:** `checkpoints/dreams_chemberta/{inflated,corrected}.pt`, the authors' checkpoints without optimizer state.

**Environment.** The locked environment (`bash environments/install.sh`; see [installation](../../../docs/install.md)). A CUDA GPU is recommended.

## Run

From the repository root:

```bash
bash paper/table3/dreams_chemberta/run.sh
```

Set `DEVICE=cuda:0` for GPU inference and `OUT=...` to choose the output directory.
`run.sh` predicts and evaluates with the fixed `table3` protocol.

To retrain:

```bash
python -m msms_models.models.retrieval.dreams_chemberta.train --config paper/table3/dreams_chemberta/train-corrected.yaml
```
