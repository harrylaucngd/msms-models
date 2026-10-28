# Table 2 · MIST with predicted or ground-truth formulas

| Formula given to MIST | Hit@1 | Hit@5 | Hit@20 | MCES@1 |
|---|---|---|---|---|
| MIST-CF top-1 | 32.14% (31.04–33.32) | 45.32% (44.08–46.51) | 60.92% (59.69–62.18) | 15.64 (15.27–16.02) |
| MIST-CF test-only top-1 | 33.55% (32.40–34.72) | 47.85% (46.65–49.15) | 62.12% (60.89–63.41) | 14.75 (14.39–15.12) |
| MIST-CF train-and-test top-1 | 36.49% (35.25–37.67) | 50.60% (49.39–51.83) | 65.60% (64.45–66.77) | 13.66 (13.32–14.01) |
| Ground truth | 52.84% (51.53–54.19) | 69.86% (68.69–71.01) | 84.17% (83.26–85.08) | 7.28 (7.03–7.54) |

Intervals are 99.9% BCa over 20,000 spectrum-level resamples.

**What runs.** The mass-based challenge: each test spectrum's candidates are all molecules
of similar mass, with no formula filter. MIST *4M-aug* encodes each spectrum with the
subformulae of the given formula. MIST-CF, trained on the standard split, on a split
contaminated with test spectra, or on train and test, supplies the top-1 formula
(Appendix A.2.3). The posteriors are thresholded at 0.187, and the candidates are ranked by
binary Tanimoto with the [`mist`](../../../src/msms_models/models/retrieval/mist.py) model.
Spectra for which no formula was available are not scored.

| Config | Posteriors (`data/paper/posteriors/`) |
|---|---|
| [`config-mistcf_standard.yaml`](config-mistcf_standard.yaml) | `mist_4m_aug_table2_mistcf_standard/` |
| [`config-mistcf_testonly.yaml`](config-mistcf_testonly.yaml) | `mist_4m_aug_table2_mistcf_testonly/` |
| [`config-mistcf_traintest.yaml`](config-mistcf_traintest.yaml) | `mist_4m_aug_table2_mistcf_traintest/` |
| [`config-ground_truth.yaml`](config-ground_truth.yaml) | `mist_4m_aug_table2_ground_truth/` |

**Inputs.** The official test queries and mass-based candidates from
`paper/prepare_data.sh`, and the posteriors from
[`harrylau/msms-models`](https://huggingface.co/harrylau/msms-models). The MIST-CF formulas
behind them are in `data/paper/table2/mist_cf_formulas.jsonl`.

## Run

From the repository root:

```bash
bash paper/table2/mist/run.sh
```

Set `OUT=...` to choose the output directory.
`run.sh` predicts and evaluates with the fixed `table2` protocol.
