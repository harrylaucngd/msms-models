# Table 2 · Random, Random w/ ground-truth formula

| | Hit@1 | Hit@5 | Hit@20 | MCES@1 |
|---|---|---|---|---|
| Random | 0.43% (0.29–0.61) | 2.01% (1.68–2.38) | 7.99% (7.34–8.74) | 30.91 (30.50–31.35) |
| Random w/ ground-truth formula | 12.98% (12.15–13.83) | 39.09% (37.94–40.31) | 71.33% (70.21–72.44) | 12.46 (12.28–12.65) |

Intervals are 99.9% BCa over 20,000 spectrum-level resamples.

**Source.** [`models/retrieval/random.py`](../../../src/msms_models/models/retrieval/random.py) ([MassSpecGym](https://github.com/pluskal-lab/MassSpecGym)).

**What runs.** The mass-based challenge: each test spectrum's candidates are all molecules of similar mass. `config-mass.yaml` ranks them by uniform random scores. `config-mass-formula.yaml` scores only the candidates whose formula equals the query's ground-truth formula, so it ranks the formula-known pool at random; the others follow. The scores come from a generator seeded with `seed`. No spectrum is read.

**Inputs.** Official MassSpecGym test queries (their formula is taken from the target) and the original mass-based candidate pool (`bash paper/prepare_data.sh`).

## Run

From the repository root:

```bash
bash paper/table2/random/run.sh
```

Set `OUT=...` to choose the output directory.
`run.sh` predicts and evaluates with the fixed `table2` protocol.
