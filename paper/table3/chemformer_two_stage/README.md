# Table 3 · (2) Chemformer two-stage retrieval (GLMR)

| Row | Hit@1 | Hit@5 | Hit@20 | MCES@1 |
|---|---|---|---|---|
| inflated | 63.89% (62.71–65.03) | 70.52% (69.37–71.70) | 75.86% (74.79–76.89) | 5.39 (5.20–5.60) |
| corrected | 7.60% (6.94–8.31) | 19.12% (18.18–20.11) | 39.31% (38.16–40.53) | 13.23 (13.06–13.39) |

Intervals are 99.9% BCa over 20,000 spectrum-level resamples.

**Source.** [`models/retrieval/chemformer_two_stage`](../../../src/msms_models/models/retrieval/chemformer_two_stage/)
ports [harrylaucngd/GLMR](https://github.com/harrylaucngd/GLMR) at `c6327e2`, the
authors' reimplementation of GLMR (arXiv:2511.06259). The networks start from the
pretrained [Chemformer](https://github.com/MolecularAI/Chemformer) BART checkpoint.

**What runs.** Stage 1 ranks the formula pool by cosine similarity between a
contrastively trained spectrum embedding and frozen Chemformer CLS embeddings. Stage
2 runs beam search (beam 5) with the Chemformer decoder on the spectrum fused with
the top 40 candidates, then re-ranks those 40 by similarity to the decoded molecule.
The two rows use the same code.

- *Inflated* is trained and evaluated on the original MassSpecGym SMILES. In that
  pool the true molecule keeps the dataset's SMILES style while the other candidates
  are written differently, and the Chemformer embeddings pick up that difference.
- *Corrected* writes every SMILES in RDKit canonical form.

**Inputs.**

| Row | Queries | Candidates |
|---|---|---|
| inflated | official test split, `data/msg/queries.jsonl` | official original formula pool, `data/msg/candidates_formula.jsonl` |
| corrected | the same queries | official MassSpecGym 1.5 formula pool, prepared by `paper/prepare_data.sh` as `data/msg/candidates_formula_1.5.jsonl` |

These inputs were checked against the files of the original runs:

| Original file | Official equivalent | Result |
|---|---|---|
| `SpecBridge_MassSpecGym_dataset.mgf` | `MassSpecGym.tsv` | Same IDs, order, folds, SMILES, precursor m/z, adduct and formula; float32 peaks equal for all 231,104 spectra; peaks are sorted by m/z |
| `_ca.mgf` | `MassSpecGym1.5.tsv` | Same spectra; SMILES are RDKit canonical |
| `SpecBridge_MSGYM_candidates.pkl` | `MassSpecGym_retrieval_candidates_formula.json` | Identical |
| `_ca.pkl` | `MassSpecGym1.5_retrieval_candidates_formula.json` | Identical on all 3,170 test targets; differs in one double-bond notation in each of two non-test lists |

Re-canonicalizing the original pool with RDKit 2025.9.5 reproduces only 2,987 of
the 3,170 test lists, so use the 1.5 file rather than canonicalizing again. No pool
holds a second candidate with the target's InChIKey-14, so the evaluator's identity
test gives the original exact-string ranks: Hit@1 from the archived top-1 is
63.89% / 7.60%.

The prepared ID-keyed files reproduce the original lists for 17,556 of 17,556
spectra. Their SHA-256 hashes:

| File | SHA-256 |
|---|---|
| `queries.jsonl` | `0d070aed…cd962` |
| `candidates_formula.jsonl` | `cd740d6e…47ad7` |
| `candidates_formula_1.5.jsonl` | `87b8d2f5…56c5e` |

**Weights.**

| Asset | Size | SHA-256 |
|---|---|---|
| `checkpoints/chemformer/step=1000000.ckpt` (Chemformer, Apache-2.0) | 224 MB | `6863bdbe…a0cb` |
| `checkpoints/chemformer_two_stage/inflated/pre_retrieval.pt` | 227 MB | `f87be96c…d3fc` |
| `checkpoints/chemformer_two_stage/inflated/generative.pt` | 321 MB | `4ba27690…69fe` |
| `checkpoints/chemformer_two_stage/corrected/pre_retrieval.pt` | 227 MB | `35ed0985…f8a7` |
| `checkpoints/chemformer_two_stage/corrected/generative.pt` | 321 MB | `5922ba4a…1366` |

The configurations pin all hashes. Publication is pending, and the GLMR code has no
license file.

**Environment.** Install `pip install -e '.[chemformer-two-stage,mces]'` on a CUDA
GPU. The original runs used PyTorch 2.5.1+cu121 and NumPy 2.2.6 (OpenBLAS 0.3.29),
most likely on an H200.

**Bootstrap.** The archived intervals come from GLMR's script. It used SciPy BCa
with a fresh `numpy.random.default_rng(42 + i)` for the i-th metric (Recall@1, @5,
@20, MRR, MCES@1), in test-set order. `msms-models evaluate --seed 42` draws from a
different random stream. On the archived ranks it reproduces every mean but moves
the bounds by up to 0.08 points (Hit@1 inflated 62.64–65.11). `run.sh` therefore
checks the printed means and reports the intervals separately.

**Reproduction status.**

- *Subset inference, verified.* The ported `predict` ran on the first 96 test
  spectra (three complete batches of 32) on an H200. For both rows, every rank,
  top-1 candidate and decoded SMILES matched the archived predictions in
  `full_test_predictions.csv` (96/96 each). Every candidate embedding (802 per row)
  was bit-identical to the original run's cache.
- *Full test set, not run.* The archived per-spectrum ranks reproduce the printed
  Hit@1/5/20 means with this evaluator.
- *MCES@1, not re-verified.* GLMR's MCES used a 600 s solver limit with a bond-count
  fallback, while the evaluator uses 5 s with 100 for failures, so MCES@1 may differ
  slightly.
- *Training.* Each released checkpoint's stored configuration is reproduced by
  `train.default_config`. On a 12-spectrum CPU run, the original scripts and the
  port produced bit-identical weights and top-K files for stage 1, the
  precomputation and both stage-2 loops. Full training was not repeated.

## Run

From the repository root:

```bash
bash paper/table3/chemformer_two_stage/run.sh
```

Set `DEVICE=cuda:0` to choose the GPU and `OUT=...` to choose the output directory.
The original full-test inference took 10–15 minutes per row on one GPU. `run.sh` predicts and evaluates both rows with the
`table3` protocol, then checks the printed means with
[`compare.py`](../../compare.py).
