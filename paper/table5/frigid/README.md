# Table 5 · FRIGID (10 rounds of refinement)

| | Acc@1 | MCES@1 | Tanimoto@1 | Acc@10 | MCES@10 | Tanimoto@10 |
|---|---|---|---|---|---|---|
| FRIGID (10 rounds of refinement) | 9.05% | 12.60 | 0.39 | 12.23% | 11.41 | 0.43 |

**Source.** [`models/de_novo/frigid`](../../../src/msms_models/models/de_novo/frigid/) with the [ICEBERG simulator](../../../src/msms_models/simulators/iceberg/).

**What runs.** Eleven rounds in total: the base round plus ten ICEBERG-guided refinement transitions, each re-masking the 8 best unique candidates 12 times with probability 0.75.

**Inputs.** As FRIGID-base, plus the ICEBERG MSG checkpoints. Refinement compares candidates with the measured test spectra. Their peaks and precursor m/z equal the official TSV, but the campaign's spectrum files recorded no collision energies. The config therefore sets `measured_collision_energy: unknown`, so every measured spectrum enters with an unknown energy, as in the paper run.

**Environment.** The locked environment (`bash environments/install.sh`; see [installation](../../../docs/install.md)). GPU required.

## Run

From the repository root:

```bash
bash paper/table5/frigid/run.sh
```

Set `DEVICE=cuda:0` for GPU inference and `OUT=...` to choose the output directory.
`run.sh` predicts and evaluates with the fixed `table5` protocol.
