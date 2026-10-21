# Table 3 · Nearest Neighbor†

| Hit@1 | Hit@5 | Hit@20 | MCES@1 |
|---|---|---|---|
| 13.13% (12.30–13.97) | 26.56% (25.49–27.69) | 44.30% (43.14–45.59) | 12.45 (12.28–12.62) |

Intervals are 99.9% BCa over 20,000 spectrum-level resamples.

**Source.** [`models/retrieval/nearest_neighbor`](../../../src/msms_models/models/retrieval/nearest_neighbor/), the `subformula_only` method of [ML_MS_analysis](https://github.com/serenaklm/ML_MS_analysis) at `6817e9b` (Khoo, arXiv:2610.02249), applied to the official MassSpecGym splits.

**What runs.** Fits a train-only sub-formula token index, selects (w, α, T, k) on all 19,429 validation spectra by mean fingerprint Jaccard (the paper selected w=2, α=2, T=0.1, k=50), then ranks the official formula candidates by Tanimoto similarity to the continuous consensus fingerprint. Stable ties; every one of the 17,556 test spectra is scored, including those without a same-formula training spectrum.

**Inputs.** Official MassSpecGym TSV and original formula candidate pool (`bash paper/prepare_data.sh`).

**Environment.** `pip install -e '.[nn,mces]'` (NumPy 1.24.4, SciPy 1.10.1, RDKit 2023.9.4). CPU only.

**Bootstrap.** seed 0, manifest order (the protocol default).

**Reproduction status.** Verified against the archived upstream run on the full test set:
- **Consensus fingerprints:** this package's agree with the archived ones to 3.6×10⁻⁷, and every spectrum selects the same 50 neighbours.
- **Hit@5, Hit@20 and their intervals:** reproduced exactly.
- **Hit@1:** a few top-1 positions are float32 near-ties. The true candidate and a decoy differ by one unit in the last place, so which one wins depends on the BLAS summation order of the machine.
  - Ranking the archived fingerprints on another machine gives 13.10% (5 spectra differ).
  - This package's own fingerprints give 13.08%.
  - The printed 13.13% came from the original node.
  - `compare.py` reports this difference rather than hiding it.

## Run

From the repository root:

```bash
bash paper/table3/nearest_neighbor/run.sh
```

Set `DEVICE=cuda:0` for GPU inference and `OUT=...` to choose the output directory.
`run.sh` predicts, evaluates with the fixed `table3` protocol and checks every printed
value with [`compare.py`](../../compare.py).
