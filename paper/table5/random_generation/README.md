# Table 5 · Random generation

| | Acc@1 | MCES@1 | Tanimoto@1 | Acc@10 | MCES@10 | Tanimoto@10 |
|---|---|---|---|---|---|---|
| Random generation [8] | 0.00% | 21.11 | 0.08 | 0.00% | 18.26 | 0.11 |

**Source.** [`models/de_novo/random_generation`](../../../src/msms_models/models/de_novo/random_generation/) ([MassSpecGym](https://github.com/pluskal-lab/MassSpecGym)).

**What runs.** For each query's formula, valence assignments of its atoms are enumerated and random molecular graphs are built over them, with neighbours and bond orders sampled from the element and bond statistics of the training molecules; ten molecules per spectrum. `run.sh` first records those statistics from the training split. The generator draws from `random` and NumPy seeded with `seed`. No spectrum is read.

**Inputs.** Official MassSpecGym training and test queries with known formulas (`bash paper/prepare_data.sh`).

## Run

From the repository root:

```bash
bash paper/table5/random_generation/run.sh
```

Set `OUT=...` to choose the output directory.
`run.sh` predicts and evaluates with the fixed `table5` protocol.
