# Table 5 · FRIGID (10 rounds of refinement)

| | Acc@1 | MCES@1 | Tanimoto@1 | Acc@10 | MCES@10 | Tanimoto@10 |
|---|---|---|---|---|---|---|
| FRIGID (10 rounds of refinement) | 9.05% | 12.60 | 0.39 | 12.23% | 11.41 | 0.43 |


**Source.** [`models/de_novo/frigid`](../../../src/msms_models/models/de_novo/frigid/) with the [ICEBERG simulator](../../../src/msms_models/simulators/iceberg/).

**What runs.** Eleven rounds in total: the base round plus ten ICEBERG-guided refinement transitions, each re-masking the 8 best unique candidates 12 times with probability 0.75.

**Inputs.** As FRIGID-base, plus the ICEBERG MSG checkpoints and test spectra with collision energies.

**Environment.** `pip install -c environments/legacy-graph.txt -e '.[frigid,iceberg,mces]'` with matching DGL and torch-scatter wheels; GPU required at full budget.

**Bootstrap.** Table 5 prints no intervals.

**Reproduction status.** Evaluator verified on the archived round-11 predictions (Acc and Tanimoto exact). The refinement inputs and campaign code still need to be checked against this port.

## Run

From the repository root:

```bash
bash paper/table5/frigid/run.sh
```

Set `DEVICE=cuda:0` for GPU inference and `OUT=...` to choose the output directory.
`run.sh` predicts, evaluates with the fixed `table5` protocol and checks every printed
value with [`compare.py`](../../compare.py).
