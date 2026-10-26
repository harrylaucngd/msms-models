# Table 3 · (2) Chemformer two-stage retrieval (GLMR)

| | Hit@1 | Hit@5 | Hit@20 | MCES@1 |
|---|---|---|---|---|
| inflated (diagnostic) | 63.89% (62.71–65.03) | 70.52% (69.37–71.70) | 75.86% (74.79–76.89) | 5.39 (5.20–5.60) |
| corrected | 7.60% (6.94–8.31) | 19.12% (18.18–20.11) | 39.31% (38.16–40.53) | 13.23 (13.06–13.39) |

Intervals are 99.9% BCa over 20,000 spectrum-level resamples.

**Source.** [`models/retrieval/chemformer_two_stage`](../../../src/msms_models/models/retrieval/chemformer_two_stage/),
a port of [harrylaucngd/GLMR](https://github.com/harrylaucngd/GLMR), the authors'
reimplementation of two-stage generative retrieval (arXiv:2511.06259). The networks start
from the pretrained [Chemformer](https://github.com/MolecularAI/Chemformer) BART checkpoint.

**What runs.** Stage 1 ranks the formula pool by cosine similarity between a contrastively
trained spectrum embedding and frozen Chemformer CLS embeddings. Stage 2 runs beam search
(beam 5) with the Chemformer decoder on the spectrum fused with the top 40 candidates, then
re-ranks those 40 by similarity to the decoded molecule. The two rows use the same code:

- **Inflated** is trained and evaluated on the original MassSpecGym SMILES, where the true
  molecule keeps the dataset's SMILES style while the other candidates are written
  differently. The Chemformer embeddings pick up that difference.
- **Corrected** writes every SMILES in RDKit canonical form (MassSpecGym 1.5).

**Inputs.**
- **Queries:** the official test spectra.
- **Candidates:** the original formula pool (inflated) or the 1.5 pool (corrected), both prepared by `paper/prepare_data.sh`.
- **Weights:** the Chemformer checkpoint plus each row's pre-retrieval and generative weights, under `checkpoints/chemformer_two_stage/{inflated,corrected}/`.

**Environment.** The locked environment (`bash environments/install.sh`; see [installation](../../../docs/install.md)). A CUDA GPU is needed; the original runs used fp16 autocast.

## Run

From the repository root:

```bash
bash paper/table3/chemformer_two_stage/run.sh
```

Set `DEVICE=cuda:0` for GPU inference and `OUT=...` to choose the output directory.
`run.sh` predicts and evaluates with the fixed `table3` protocol.

Training (both stages and the top-40 precomputation) is in [`train.py`](../../../src/msms_models/models/retrieval/chemformer_two_stage/train.py);
see the [method guide](../../../src/msms_models/models/retrieval/chemformer_two_stage/README.md).
