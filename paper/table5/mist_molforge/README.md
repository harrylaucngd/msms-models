# Table 5 · MIST *4M-aug* + MolForge† (corrected)

| | Acc@1 | MCES@1 | Tanimoto@1 | Acc@10 | MCES@10 | Tanimoto@10 |
|---|---|---|---|---|---|---|
| MIST + MolForge (corrected) | 3.29% | 31.77 | 0.31 | 5.65% | 16.67 | 0.40 |

**Source.** [`models/de_novo/molforge`](../../../src/msms_models/models/de_novo/molforge/) ([knu-lcbc/MolForge](https://github.com/knu-lcbc/MolForge) decoder, CC BY-NC-SA 4.0), conditioned on [`encoders/mist`](../../../src/msms_models/encoders/mist/).

**What runs.** Thresholds MIST 4M-aug posteriors at 0.18791015 and decodes ten beams per spectrum. Raw beam slots are kept in order: invalid strings count as misses (Tanimoto 0, MCES 100) and nothing is filtered or backfilled.

**Inputs.** Official MassSpecGym test queries with known formulas; the released MIST 4M-aug Table 5 posterior cache; MolForge decoder and its two SentencePiece models.

## Run

From the repository root:

```bash
bash paper/table5/mist_molforge/run.sh
```

Set `DEVICE=cuda:0` for GPU inference and `OUT=...` to choose the output directory.
`run.sh` predicts and evaluates with the fixed `table5` protocol.
