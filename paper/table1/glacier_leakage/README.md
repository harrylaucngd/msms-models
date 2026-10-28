# Table 1 · Leakage from synthetic spectrum augmentation

| # | GLACIER split | Augmented data source | MIST spec2fp Tanimoto | Hit@1 |
|---|---|---|---|---|
| 1 | MSG train + val. + test | MSG-4M + MSG test | 0.475 | 47.01% |
| 2 | MSG train + val. + test | MSG-4M | 0.449 | 43.42% |
| 3 | MSG train | MSG-4M + MSG test | 0.447 | 43.16% |
| 4 | MSG train | MSG-4M | 0.433 | 40.31% |
| 5 | MIST (no augmentation) | | 0.262 | 9.57% |

Each setting is a MIST encoder trained on the MassSpecGym training spectra mixed with one
GLACIER augmentation corpus (Appendix A.2); setting 4 is MIST *4M-aug*, the encoder of the
[Table 3 row](../../table3/mist_4m_aug/).

**What runs.** For each setting, the encoder's test posteriors are thresholded at its
validation-calibrated threshold and the formula-based candidates are ranked by binary
Tanimoto (the [`mist`](../../../src/msms_models/models/retrieval/mist.py) model). The
spec2fp Tanimoto compares the thresholded posterior with the target's Morgan radius-2/4096
fingerprint ([`evaluation/fingerprints.py`](../../../src/msms_models/evaluation/fingerprints.py)).

| Config | Posteriors (`data/paper/posteriors/`) |
|---|---|
| [`config-setting1.yaml`](config-setting1.yaml) | `mist_table1_setting1/` |
| [`config-setting2.yaml`](config-setting2.yaml) | `mist_table1_setting2/` |
| [`config-setting3.yaml`](config-setting3.yaml) | `mist_table1_setting3/` |
| [`config-setting4.yaml`](config-setting4.yaml) | `mist_4m_aug_table3/`, on the Table 3 inputs |
| [`config-setting5.yaml`](config-setting5.yaml) | `mist_noaug_table1/` |

**Inputs.** The official test queries and formula candidates from `paper/prepare_data.sh`,
and the posteriors from [`harrylau/msms-models`](https://huggingface.co/harrylau/msms-models).

## Run

From the repository root:

```bash
bash paper/table1/glacier_leakage/run.sh
```

Each setting writes `metrics.json` (retrieval, fixed `table3` protocol) and
`fingerprints.json` (spec2fp Tanimoto). Set `OUT=...` to choose the output directory.
