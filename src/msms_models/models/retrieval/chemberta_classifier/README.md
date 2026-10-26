# ChemBERTa SMILES classifier

A spectrum-blind baseline (paper Appendix A.4, method 3). [`train.py`](train.py) fine-tunes
[DeepChem/ChemBERTa-77M-MLM](https://huggingface.co/DeepChem/ChemBERTa-77M-MLM) with a
two-class head on balanced pairs from the training split: each molecule's SMILES (label 1)
and one decoy drawn from its candidate list (label 0). [`inference.py`](inference.py) scores
every candidate string once and ranks each pool by the probability of label 1. No spectrum
is read. [`model.py`](model.py) holds the loader and the SMILES canonicalization.

## Assets

| Configuration key | Input |
|---|---|
| `classifier` | A directory written by `train.py`: weights, tokenizer and `training.json` |
| `candidates` | ID-keyed formula candidate pool |

**Parameters.** `batch_size` (256) is the number of strings scored per batch. Whether
SMILES are RDKit-canonicalized comes from `canonical_smiles` in `<classifier>/training.json`,
which `train.py` writes (`--canonical` sets it); the maximum length (256) comes from there too.
Training defaults: 2 epochs, batch size 32, learning rate 5e-5, seed 0.

## Run

From the repository root, after `bash paper/prepare_data.sh`:

```bash
bash environments/install.sh   # from the repository root
python -m msms_models.models.retrieval.chemberta_classifier.train --canonical \
  --queries data/msg/train.jsonl --candidates data/MassSpecGym_retrieval_candidates_formula.json \
  --output checkpoints/chemberta_classifier/corrected
msms-models predict --config paper/table3/chemberta_classifier/config-corrected.yaml
msms-models evaluate --protocol table3 \
  --queries results/paper/table3/chemberta_classifier_corrected/queries.jsonl \
  --predictions results/paper/table3/chemberta_classifier_corrected/predictions.jsonl \
  --output results/paper/table3/chemberta_classifier_corrected/metrics.json
```

Drop `--canonical` and use `config-inflated.yaml` for the inflated row.
[`paper/table3/chemberta_classifier/run.sh`](../../../../../paper/table3/chemberta_classifier/run.sh)
trains and evaluates both rows.
