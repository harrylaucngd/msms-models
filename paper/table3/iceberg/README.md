# Table 3 · ICEBERG (forward-model retrieval)

| Hit@1 | Hit@5 | Hit@20 | MCES@1 |
|---|---|---|---|
| 36.18% (34.96–37.32) | 60.70% (59.58–61.86) | 78.83% (77.74–79.78) | 6.61 (6.43–6.79) |

Intervals are 99.9% BCa over 20,000 spectrum-level resamples.

**Source.** [`simulators/iceberg`](../../../src/msms_models/simulators/iceberg/) ([coleygroup/ms-pred](https://github.com/coleygroup/ms-pred)) with [`models/retrieval/iceberg.py`](../../../src/msms_models/models/retrieval/iceberg.py).

**What runs.** Predicts a spectrum for every candidate at the query's collision energy and ranks candidates by native entropy distance to the measured spectrum. About half of the test spectra lack a collision energy; the frozen queries carry the energies imputed as in Appendix A.5.

**Inputs.** Frozen Table 3 queries (with imputed collision energies) and candidate pool; ICEBERG MSG generator/intensity checkpoints.

**Environment.** `pip install -c environments/legacy-graph.txt -e '.[iceberg,mces]'` with matching DGL and torch-scatter wheels. GPU recommended.

**Bootstrap.** seed 1, `target-rank` order assumed (as for GLACIER); not recorded in an archived result.

**Reproduction status.** **Not yet verified.** The archived ICEBERG retrieval outputs were not found, so this row needs a full rerun to confirm 36.18%.

## Run

From the repository root:

```bash
bash paper/table3/iceberg/run.sh
```

Set `DEVICE=cuda:0` for GPU inference and `OUT=...` to choose the output directory.
`run.sh` predicts, evaluates with the fixed `table3` protocol and checks every printed
value with [`compare.py`](../../compare.py).
