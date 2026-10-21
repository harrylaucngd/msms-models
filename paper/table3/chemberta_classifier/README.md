# Table 3 · (3) ChemBERTa SMILES classifier

| | Hit@1 | Hit@5 | Hit@20 | MCES@1 |
|---|---|---|---|---|
| (3) ChemBERTa SMILES Classifier (inflated) | 99.66% (99.50–99.79) | 99.84% (99.72–99.92) | 99.99% (99.95–100.00) | 0.05 (0.03–0.07) |
| (3) ChemBERTa SMILES Classifier (corrected) | 8.20% (7.53–8.89) | 18.39% (17.38–19.36) | 38.14% (36.96–39.36) | 13.80 (13.64–13.96) |

Intervals are 99.9% BCa over 20,000 spectrum-level resamples.

**Source.** [`models/retrieval/chemberta_classifier`](../../../src/msms_models/models/retrieval/chemberta_classifier/) (`train.py`, `inference.py`) on [DeepChem/ChemBERTa-77M-MLM](https://huggingface.co/DeepChem/ChemBERTa-77M-MLM).

**What runs.** Fine-tunes ChemBERTa with a randomly initialized two-class head for two epochs on balanced pairs: each training molecule's SMILES (label 1) and one decoy sampled from its candidate list (label 0). At test time every candidate is scored independently and ranked by its probability of being a query molecule; no spectrum is read. *Inflated* uses the original strings, where true molecules keep their repository SMILES syntax. *Corrected* RDKit-canonicalizes all queries and candidates, as MassSpecGym 1.5 does. The learning rate (5e-5), batch size (32) and maximum length (256) are this implementation's choices; the paper does not state them.

**Inputs.** Official training queries and test queries with the original formula pool. `run.sh` trains each classifier if `checkpoints/chemberta_classifier/<variant>` is absent.

**Environment.** `pip install -e '.[chemberta,mces]'`; a GPU is recommended for training.

## Run

From the repository root:

```bash
bash paper/table3/chemberta_classifier/run.sh
```

Set `DEVICE=cuda:0` for GPU inference and `OUT=...` to choose the output directory.
`run.sh` predicts and evaluates with the fixed `table3` protocol.
