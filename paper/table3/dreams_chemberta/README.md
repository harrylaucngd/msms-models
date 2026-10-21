# Table 3 · (1) DreaMS + ChemBERTa alignment (SpecBridge)

| Row | Hit@1 | Hit@5 | Hit@20 | MCES@1 |
|---|---|---|---|---|
| inflated (diagnostic) | 82.41% (81.46–83.33) | 92.57% (91.91–93.21) | 97.41% (97.00–97.78) | 2.86 (2.70–3.03) |
| corrected | 6.70% (6.09–7.35) | 19.82% (18.89–20.83) | 40.62% (39.44–41.79) | 13.13 (12.97–13.28) |

Intervals are 99.9% BCa over 20,000 spectrum-level resamples, n = 17,556.

**Source.** [`models/retrieval/dreams_chemberta`](../../../src/msms_models/models/retrieval/dreams_chemberta/),
a port of [HassounLab/SpecBridge](https://github.com/HassounLab/SpecBridge) (`6c6fadf` with
the authors' local evaluation changes) and the authors' scripts
`benchmark_msgym/compute_bootstrap_mces_msgym_retrained2.py` (inflated) and
`run_msgym_augmented_pubchem13m_ca_bootstrap.py` (corrected). DreaMS layers come from
[pluskal-lab/DreaMS](https://github.com/pluskal-lab/DreaMS) `02be141a2dd1`.

**What runs.** DreaMS embeds each spectrum. The masked mean of its peak embeddings is
projected and L2-normalised, then a residual mapper places it in the CLS space of a frozen
ChemBERTa. Candidates are ranked by the dot product with their ChemBERTa CLS embeddings.
The two rows share code and settings and differ only in their inputs:

- **Inflated.** The raw MassSpecGym strings. The true SMILES keeps its PubChem (Kekulé)
  spelling, while the decoys are RDKit-canonical. The alignment checkpoint uses a ChemBERTa
  further MLM-trained on canonical SMILES (test InChIKey-14s excluded), so the string
  style identifies the answer.
- **Corrected.** Every query and candidate string is RDKit-canonicalised (SpecBridge
  `_ca` files), with the public ChemBERTa.

Both checkpoints were trained for 2 epochs with batch size 32, lr 1e-4, seed 1234,
`5·MSE + 1e-6·orthogonality`, the last 2 DreaMS blocks unfrozen and ChemBERTa frozen.
The training configs are [`train-inflated.yaml`](train-inflated.yaml) and
[`train-corrected.yaml`](train-corrected.yaml). Retraining is not needed for this row.

**Inputs.**

- **Queries.** The official test spectra, `data/msg/queries.jsonl` from `prepare_data.sh`.
  SpecBridge's MGF holds exactly these spectra (17,556; identical m/z and intensity
  values, same order), and its `_ca` MGF only canonicalises the SMILES, which inference
  never reads.
- **Inflated candidates.** SpecBridge's raw pool equals the official
  `MassSpecGym_retrieval_candidates_formula.json` (all 32,010 lists, order included).
  `run.sh` rebuilds the candidate file from that pool.
- **Corrected candidates.** SpecBridge's `_ca` pool equals the official
  `MassSpecGym1.5_retrieval_candidates_formula.json`. Both have the same 32,010 keys in the same
  order and identical test pools, keyed by the RDKit-canonical target. `run.sh` rebuilds the file from it with
  `--canonical-keys`. Building it from `SpecBridge_MSGYM_candidates_ca.pkl` gives the same
  SHA-256. Canonicalising the original pool with RDKit 2025.09.5 instead differs in 215
  double-bond-stereo spellings across 183 test pools, so use the 1.5 pool.
- **Candidate order.** Both files store upstream's evaluation order, produced by
  [`candidates.py`](../../../src/msms_models/models/retrieval/dreams_chemberta/candidates.py):
  `random.seed(1234)`, one shuffle per pool in file order. The official pools list the
  true molecule first.

| Input | Path | SHA-256 |
|---|---|---|
| inflated weights (inference-only `last.pt`) | `checkpoints/dreams_chemberta/inflated.pt` | `65de2033…ce276` |
| corrected weights | `checkpoints/dreams_chemberta/corrected.pt` | `8d5ee7fd…ec398` |
| inflated candidates | `data/paper/table3/dreams_chemberta/candidates_inflated.jsonl` | `f4395714…e031c` |
| corrected candidates | `data/paper/table3/dreams_chemberta/candidates_corrected.jsonl` | `254f82d5…c509` |

The weights are the original `last.pt` files without optimizer state, written by
`model.convert_checkpoint`. Every tensor, ChemBERTa included, loads strictly. The original
files have SHA-256 `e7c7adde…0fca55` (inflated) and `570eb823…428ba` (corrected).

**Environment.** `pip install -e '.[dreams-chemberta,mces]'`. The paper ran with
Python 3.11, torch 2.3.0, transformers 5.0.0, tokenizers 0.22.2 and numpy 1.24.4. The
ChemBERTa tokenizer and config are pinned to revision `2da3150903b3`. The archived run used
a GPU; CPU works, but expect float differences around 1e-5 (see below).

**Bootstrap.** The archived intervals used
`scipy.stats.bootstrap(method="BCa", confidence_level=0.999, n_resamples=20000)` in query
order, with `random_state=np.random.default_rng(seed)`. The seed is
`20260506 + 1000·r + m`, where r = 0 for inflated and 1 for corrected, and m is the metric
index: R@1 = 0, R@5 = 1, R@10 = 2, R@20 = 3, MRR = 4, MCES = 5.

The package evaluator draws one `RandomState` stream for all metrics, so it cannot
reproduce these bounds. `run.sh` therefore checks means only (`--bootstrap-resamples 0`).

The archived MCES@1 also differs from the package protocol:
- It used a 600 s solver limit instead of 5 s.
- It took the mean over finite distances.
- Identity was decided by string rank, not by InChIKey-14.

**Reproduction status.**

- **Verified on a subset.** The ported `predict` ran on the first 96 test queries (three
  full batches of 32) with each converted checkpoint, on CPU.
  - The rank of the true candidate matched the archived per-query rank for **96/96 queries
    in both rows**. Hit@1/5/20 on that subset are 62.5/97.9/100% (inflated) and
    0/15.6/63.5% (corrected), identical to the archive.
  - The package evaluator (InChIKey-14) gives the same values.
  - CPU candidate embeddings differ from the archived GPU cache by at most 1.2e-5.
  - ChemBERTa gives stereoisomers exactly tied scores. The top-1 *string* therefore
    differed in 9 (inflated) and 5 (corrected) queries. Its tokenizer drops bracket atoms with
    `@`, so `[C@@H]` and `[C@H]` both become `C`. In every case the archived top-1
    sat in the same tied group, which shares the InChIKey-14 and achiral structure.
- **Training.** A 4-step smoke run (64 training spectra, CPU) of the ported `train.py`
  gave the same losses and bitwise-identical weights (all 231 tensors) as upstream
  `specbridge.train.train` for both variants. The paper training configs reproduce every
  stored `args` value that affects training.
- **Not run.** Full-test inference, MCES and intervals. Compute was limited, so
  `compare.py` has not been run against the printed rows.

## Run

From the repository root:

```bash
bash paper/table3/dreams_chemberta/run.sh
```

Set `DEVICE=cuda:0` for GPU inference. `OUT=...` chooses the parent directory of the two
row outputs. `run.sh` does three things for each row:
1. Predicts.
2. Evaluates with the fixed `table3` protocol.
3. Checks the printed means with [`compare.py`](../../compare.py).
