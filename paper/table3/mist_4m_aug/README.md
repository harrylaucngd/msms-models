# Table 3 · MIST *4M-aug*

| Hit@1 | Hit@5 | Hit@20 | MCES@1 |
|---|---|---|---|
| 40.31% (39.05–41.47) | 58.87% (57.66–60.06) | 73.02% (71.95–74.11) | 7.62 (7.44–7.82) |

Intervals are 99.9% BCa over 20,000 spectrum-level resamples.

**Source.** [`models/retrieval/mist.py`](../../../src/msms_models/models/retrieval/mist.py) and [`models/retrieval/fingerprint.py`](../../../src/msms_models/models/retrieval/fingerprint.py); encoder [`encoders/mist`](../../../src/msms_models/encoders/mist/) ([coleygroup/MIST](https://github.com/coleygroup/MIST), checkpoint `548c1301…`, EMA weights).

**What runs.** Thresholds the MIST 4M-aug posteriors at the validation-calibrated 0.18791085 and ranks the frozen Table 3 formula pool by binary Tanimoto, breaking ties with NumPy 1.24.4's reverse quicksort as in the original MIST evaluation (stable ties give 40.82% Hit@1).

**Inputs.** Frozen Table 3 queries and candidate pool, and the released MIST 4M-aug Table 3 posterior cache (see [paper/README.md](../../README.md#inputs-and-assets)).

**Environment.** `pip install -e '.[mces]'` with NumPy 1.24.4. Ranking from the cache needs no PyTorch.

## Run

From the repository root:

```bash
bash paper/table3/mist_4m_aug/run.sh
```

Set `DEVICE=cuda:0` for GPU inference and `OUT=...` to choose the output directory.
`run.sh` predicts and evaluates with the fixed `table3` protocol.
