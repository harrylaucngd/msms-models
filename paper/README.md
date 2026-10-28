# Reproducing *MassSpecGym in the Wild*

Every method entry of the paper's main text is either **implemented here** or **cited**
from its authors. Each implemented entry has one folder with its configurations, a
`run.sh` that predicts and evaluates, and a README stating what runs and what it needs.
Method source lives once, in [`src/msms_models`](../src/msms_models/); the folders bind
it to the paper's inputs and settings.

## Running an entry

```bash
bash environments/install.sh && source .venv/bin/activate   # one environment for every entry
bash paper/prepare_data.sh                                  # official MassSpecGym inputs
hf download harrylau/msms-models --local-dir . --include 'checkpoints/*' 'data/*'
bash paper/table3/mist_4m_aug/run.sh                        # any entry below
```

The Hugging Face download adds the checkpoints and frozen inputs under `checkpoints/` and
`data/`. Every `run.sh` runs from the repository root and writes `metrics.json` next to
the predictions. `OUT=...` chooses the output directory and `DEVICE=cuda:0` the device.
Entries that train a model first, or need inputs beyond these three steps, say so in the
**Preparation** column and in their README.

## Table 1 · Leakage from synthetic spectrum augmentation

| Rows | Folder | Preparation |
|---|---|---|
| Settings 1–5 (MIST spec2fp Tanimoto, Hit@1) | [table1/glacier_leakage](table1/glacier_leakage/) | — |

## Table 2 · Retrieval on mass-based candidates

| Rows | Folder | Preparation |
|---|---|---|
| Random, Random w/ ground-truth formula | [table2/random](table2/random/) | — |
| MIST with MIST-CF top-1 formulas (standard, test-only, train-and-test) and with ground-truth formulas | [table2/mist](table2/mist/) | — |

## Table 3 · Retrieval with known formula

| Rows | Folder | Preparation |
|---|---|---|
| Random | [table3/random](table3/random/) | — |
| Nearest Neighbor† | [table3/nearest_neighbor](table3/nearest_neighbor/) | builds its index (`run.sh`) |
| DeepSets, DeepSets + Fourier features | [table3/deepsets](table3/deepsets/) | trains (`run.sh`) |
| Fingerprint FFN | [table3/fingerprint_ffn](table3/fingerprint_ffn/) | trains (`run.sh`) |
| MIST vMassSpecGym (no augmentation) | [table3/mist_vmassspecgym](table3/mist_vmassspecgym/) | — |
| MIST *4M-aug* | [table3/mist_4m_aug](table3/mist_4m_aug/) | — |
| ICEBERG | [table3/iceberg](table3/iceberg/) | GPU |
| GLACIER | [table3/glacier](table3/glacier/) | GPU |
| (1) DreaMS + ChemBERTa Alignment, inflated / corrected | [table3/dreams_chemberta](table3/dreams_chemberta/) | builds its candidate files (`run.sh`) |
| (2) Chemformer Two-Stage Retrieval, inflated / corrected | [table3/chemformer_two_stage](table3/chemformer_two_stage/) | — |
| (3) ChemBERTa SMILES Classifier, inflated / corrected | [table3/chemberta_classifier](table3/chemberta_classifier/) | trains (`run.sh`) |
| (4) PubChem Default Ranking | [table3/pubchem_order](table3/pubchem_order/) | queries PubChem (`run.sh`) |
| (5) Ranking by Number of Chiral Atoms | [table3/chiral_atoms](table3/chiral_atoms/) | — |

## Table 4 · MIST + MolForge with correct and problematic encoders

| Rows | Folder | Preparation |
|---|---|---|
| MIST *4M-aug* + MolForge† | [table5/mist_molforge](table5/mist_molforge/) | GPU |
| MIST vDiffMS + MolForge† (batch size 1, batch size 24) | [table5/mist_vdiffms_molforge](table5/mist_vdiffms_molforge/) | GPU |

Both folders also write the encoder's MIST Tanimoto (`fingerprints.json`).

## Table 5 · De novo generation with known formula

| Rows | Folder | Preparation |
|---|---|---|
| SMILES Transformer, SELFIES Transformer | [table5/smiles_transformer](table5/smiles_transformer/) | trains (`run.sh`), GPU |
| Random generation | [table5/random_generation](table5/random_generation/) | fits element statistics (`run.sh`) |
| MIST + MSNovelist | [table5/mist_msnovelist](table5/mist_msnovelist/) | trains the decoder (README), GPU |
| FOAM (DiffMS + PubChem) | [table5/foam](table5/foam/) | builds the seeds (README), GPU |
| MIST + MolForge† (corrected) | [table5/mist_molforge](table5/mist_molforge/) | GPU |
| MIST + MolForge† (inflated, batch size = 24) | [table5/mist_vdiffms_molforge](table5/mist_vdiffms_molforge/) | GPU |
| FRIGID-base (MIST 4M-aug) | [table5/frigid_base](table5/frigid_base/) | GPU |
| FRIGID (MIST 4M-aug) | [table5/frigid](table5/frigid/) | GPU |

## Figure 2 · Decoder querying

| Entry | Folder | Preparation |
|---|---|---|
| Querying a molecule decoder with MIST *4M-aug* posteriors | [figure2/ms_gpt](figure2/ms_gpt/) | downloads MS-GPT (`run.sh`), GPU |

The pretraining sets of the 18 Figure 2 decoders are in
[`harrylau/ms-gpt-18`](https://huggingface.co/datasets/harrylau/ms-gpt-18).

## Cited entries

These rows are reported from their authors' papers and are not implemented here.

| Method | Table | Code |
|---|---|---|
| JESTR | 3 | [HassounLab/JESTR1](https://github.com/HassounLab/JESTR1) |
| MVP | 3 | [HassounLab/MVP](https://github.com/HassounLab/MVP) |
| FLARE | 3 | [HassounLab/FLARE](https://huggingface.co/spaces/HassounLab/FLARE) (Hugging Face Space) |
| Spec2Mol | 5 | [KavrakiLab/Spec2Mol](https://github.com/KavrakiLab/Spec2Mol) |
| MADGEN | 5 | [HassounLab/MADGEN](https://github.com/HassounLab/MADGEN) |
| MS-BART | 5 | [OpenDFM/MS-BART](https://github.com/OpenDFM/MS-BART) |
| DiffMS | 5 | [coleygroup/DiffMS](https://github.com/coleygroup/DiffMS) |
| MSAnchor | 5 | no public code; [paper](https://doi.org/10.1609/aaai.v40i2.37064) |
| FRIGID (arXiv v2) | 5 | [coleygroup/FRIGID](https://github.com/coleygroup/FRIGID) |

## Inputs and assets

`prepare_data.sh` downloads the official MassSpecGym release and prepares the test
queries, the formula and mass-based candidate pools and the training and validation
splits. Everything else comes from [`harrylau/msms-models`](https://huggingface.co/harrylau/msms-models):
model weights under `checkpoints/`, and under `data/paper/`:

| Input | Used by |
|---|---|
| `table3/queries.jsonl`, `table3/candidates_formula.jsonl`: test spectra with imputed collision energies (Appendix A.5) and the formula pool in forward-model order | Table 3 MIST *4M-aug*, ICEBERG, GLACIER; Table 1 setting 4 |
| `posteriors/mist_table1_setting{1,2,3}/`, `posteriors/mist_noaug_table1/` | Table 1 |
| `posteriors/mist_4m_aug_table2_*/`, `table2/mist_cf_formulas.jsonl` | Table 2 MIST |
| `posteriors/mist_4m_aug_table3/` | Table 3 MIST *4M-aug*; Table 1 setting 4 |
| `posteriors/mist_4m_aug_table5/` | Tables 4–5 MolForge and FRIGID; Figure 2 |
| `table3/mist_vmassspecgym/subformulae.jsonl` | Table 3 MIST vMassSpecGym |
| `table5/mist_vdiffms/subformulae.jsonl` | Tables 4–5 vDiffMS rows, MIST + MSNovelist |

See [assets](../docs/assets.md) for every file.

## Evaluation protocols

`msms-models evaluate --protocol table2|table3|table5` fixes the MassSpecGym metrics
(`table2` is `table3` on the mass-based candidates):
- **Identity:** InChIKey-14.
- **Similarity:** Morgan radius-2/2048 Tanimoto without chirality.
- **MCES:** myopic MCES with threshold 15 and the stronger bound. Invalid or missing predictions count as 100.
- **Intervals:** 99.9% BCa over 20,000 spectrum-level resamples.

`msms-models evaluate-fingerprints` gives the encoder metric of Tables 1 and 4: the mean
Tanimoto between the thresholded posterior and the target's Morgan radius-2/4096
fingerprint. See [`evaluation/`](../src/msms_models/evaluation/).
