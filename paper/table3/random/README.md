# Table 3 · Random

| | Hit@1 | Hit@5 | Hit@20 | MCES@1 |
|---|---|---|---|---|
| Random [8] | 3.06% (2.64–3.52) | 11.35% (10.60–12.12) | 27.74% (26.52–28.84) | 13.87 (13.70–14.03) |

Intervals are 99.9% BCa over 20,000 spectrum-level resamples.

**Source.** [`models/retrieval/random.py`](../../../src/msms_models/models/retrieval/random.py), [MassSpecGym](https://github.com/pluskal-lab/MassSpecGym)'s `RandomRetrieval`. The printed row is MassSpecGym's.

**What runs.** Each test spectrum's formula-known candidates are ranked by uniform random scores from a seeded generator, so the numbers vary with `seed` within their intervals. No spectrum is read.

**Inputs.** Official MassSpecGym test queries and original formula candidate pool (`bash paper/prepare_data.sh`).

**Environment.** The locked environment (`bash environments/install.sh`; see [installation](../../../docs/install.md)). CPU only.

## Run

From the repository root:

```bash
bash paper/table3/random/run.sh
```

Set `OUT=...` to choose the output directory.
`run.sh` predicts and evaluates with the fixed `table3` protocol.
