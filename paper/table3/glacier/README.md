# Table 3 · GLACIER (forward-model retrieval)

| Hit@1 | Hit@5 | Hit@20 | MCES@1 |
|---|---|---|---|
| 39.47% (38.20–40.65) | 63.50% (62.35–64.66) | 80.46% (79.44–81.41) | 6.53 (6.34–6.72) |

Intervals are 99.9% BCa over 20,000 spectrum-level resamples.

**Source.** [`simulators/glacier`](../../../src/msms_models/simulators/glacier/) (architecture from the pinned [coleygroup/MIST](https://github.com/coleygroup/MIST) package) with [`models/retrieval/glacier.py`](../../../src/msms_models/models/retrieval/glacier.py).

**What runs.** As for ICEBERG, with GLACIER's max-pooled top-k binned predictions. The checkpoint is the GLACIER retrained on the collision-energy-completed MSG training split (Appendix A.5).

**Inputs.** Frozen Table 3 queries (with imputed collision energies) and candidate pool; the retrained GLACIER checkpoint.

**Environment.** The locked environment (`bash environments/install.sh`; see [installation](../../../docs/install.md)). GPU recommended.

## Run

From the repository root:

```bash
bash paper/table3/glacier/run.sh
```

Set `DEVICE=cuda:0` for GPU inference and `OUT=...` to choose the output directory.
`run.sh` predicts and evaluates with the fixed `table3` protocol.
