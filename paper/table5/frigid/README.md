# Table 5 · FRIGID (10 rounds of refinement)

| | Acc@1 | MCES@1 | Tanimoto@1 | Acc@10 | MCES@10 | Tanimoto@10 |
|---|---|---|---|---|---|---|
| FRIGID (10 rounds of refinement) | 9.05% | 12.60 | 0.39 | 12.23% | 11.41 | 0.43 |

**Source.** [`models/de_novo/frigid`](../../../src/msms_models/models/de_novo/frigid/) with the [ICEBERG simulator](../../../src/msms_models/simulators/iceberg/).

**What runs.** Eleven rounds in total: the base round plus ten ICEBERG-guided refinement transitions, each re-masking the 8 best unique candidates 12 times with probability 0.75.

**Inputs.** As FRIGID-base, plus the ICEBERG MSG checkpoints. Refinement compares candidates with the measured test spectra; the config sets `measured_collision_energy: unknown`, so every measured spectrum enters with an unknown collision energy.

## Run

From the repository root:

```bash
DEVICE=cuda:0 bash paper/table5/frigid/run.sh
```

Set `OUT=...` to choose the output directory.
`run.sh` predicts and evaluates with the fixed `table5` protocol.
