# Table 3 · Nearest Neighbor†

| Hit@1 | Hit@5 | Hit@20 | MCES@1 |
|---|---|---|---|
| 13.13% (12.30–13.97) | 26.56% (25.49–27.69) | 44.30% (43.14–45.59) | 12.45 (12.28–12.62) |

Intervals are 99.9% BCa over 20,000 spectrum-level resamples.

**Source.** [`models/retrieval/nearest_neighbor`](../../../src/msms_models/models/retrieval/nearest_neighbor/), the `subformula_only` method of [ML_MS_analysis](https://github.com/serenaklm/ML_MS_analysis) at `6817e9b` (Khoo, arXiv:2610.02249), applied to the official MassSpecGym splits.

**What runs.** Fits a train-only sub-formula token index, selects (w, α, T, k) on all 19,429 validation spectra by mean fingerprint Jaccard (the paper selected w=2, α=2, T=0.1, k=50), then ranks the official formula candidates by Tanimoto similarity to the continuous consensus fingerprint. Stable ties; every one of the 17,556 test spectra is scored, including those without a same-formula training spectrum.

**Inputs.** Official MassSpecGym TSV and original formula candidate pool (`bash paper/prepare_data.sh`).

**Environment.** `pip install -e '.[nn,mces]'` (NumPy 1.24.4, SciPy 1.10.1, RDKit 2023.9.4). CPU only.

## Run

From the repository root:

```bash
bash paper/table3/nearest_neighbor/run.sh
```

Set `DEVICE=cuda:0` for GPU inference and `OUT=...` to choose the output directory.
`run.sh` predicts and evaluates with the fixed `table3` protocol.
