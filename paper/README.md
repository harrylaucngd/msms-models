# Reproducing *MassSpecGym in the Wild*

Every method the authors ran for Tables 3 and 5 has one folder here. Each folder holds
the exact configuration, a `run.sh` that predicts, evaluates and checks the result,
and a README stating where the numbers come from and how far they have been
verified. Method source lives once, in [`src/msms_models`](../src/msms_models/); the
folders only bind it to the paper's inputs and settings.

```text
paper/
  results/table3.csv, table5.csv   every printed value, with its origin and folder
  compare.py                       checks a metrics.json against the printed row
  prepare_data.sh                  official MassSpecGym inputs
  table3/<row>/                    README.md, config.yaml, run.sh
  table5/<row>/
```

## Table 3 · Retrieval with known formula

| Row | Folder | Origin | Status |
|---|---|---|---|
| Nearest Neighbor† | [nearest_neighbor](table3/nearest_neighbor/) | reimplementation | archived values match; package rerun pending |
| MIST *4M-aug* | [mist_4m_aug](table3/mist_4m_aug/) | author rerun | **verified exactly**, intervals included |
| ICEBERG | [iceberg](table3/iceberg/) | author rerun | archived outputs missing; needs a rerun |
| GLACIER | [glacier](table3/glacier/) | author rerun | evaluator verified exactly on the archived ranking |
| (1) DreaMS + ChemBERTa, inflated / corrected | `table3/dreams_chemberta` | author experiment | being ported |
| (2) Chemformer two-stage, inflated / corrected | `table3/chemformer_two_stage` | author experiment | being ported |
| (3) ChemBERTa SMILES classifier, inflated / corrected | `table3/chemberta_classifier` | author experiment | to be rewritten from Appendix A.4 |
| (4) PubChem default ranking | `table3/pubchem_order` | spectrum-blind baseline | to be rewritten from Appendix A.4 |
| (5) Ranking by number of chiral atoms | `table3/chiral_atoms` | spectrum-blind baseline | to be rewritten from Appendix A.4 |

## Table 5 · De novo generation with known formula

| Row | Folder | Origin | Status |
|---|---|---|---|
| MIST + MolForge† (corrected) | [mist_molforge](table5/mist_molforge/) | reimplementation | **evaluator verified exactly** on archived predictions |
| MIST + MolForge† (inflated, batch size 24; Table 4) | `table5/mist_vdiffms_molforge` | diagnostic | to be added |
| FRIGID-base | [frigid_base](table5/frigid_base/) | author rerun | evaluator verified exactly on archived predictions |
| FRIGID (10 rounds of refinement) | [frigid](table5/frigid/) | author rerun | evaluator verified exactly on archived predictions |

Rows marked *cited* in [`results/`](results/) were copied from the original papers
or the [MassSpecGym leaderboard](https://github.com/pluskal-lab/MassSpecGym) and are
not implemented here. That includes FRIGID with MIST vFRIGID, which reports the
FRIGID authors' number.

## Running a row

```bash
python -m pip install -e '.[mces]'        # plus the extras named in the row's README
bash paper/prepare_data.sh
bash paper/table3/mist_4m_aug/run.sh
```

Every `run.sh` is run from the repository root and ends with
`python paper/compare.py <table> <row> <metrics.json>`. That check rounds the reproduced
means and interval bounds to the printed precision and fails on any difference.

## Evaluation protocols

`msms-models evaluate --protocol table3|table5` fixes the MassSpecGym metrics:
- **Identity:** InChIKey-14.
- **Similarity:** Morgan radius-2/2048 Tanimoto without chirality.
- **MCES:** myopic MCES with threshold 15 and the stronger bound. Invalid or missing predictions count as 100.
- **Intervals:** 99.9% BCa over 20,000 spectrum-level resamples.

Interval bounds also depend on the resampling seed and the order of spectra, so each
row's README gives the settings of its archived run. See
[`evaluation/protocols.py`](../src/msms_models/evaluation/protocols.py).

## Inputs and assets

`prepare_data.sh` prepares the official MassSpecGym TSV and formula candidate pool.
Some rows also need frozen inputs that cannot be rebuilt from the official release.
They are placed under `data/paper/`:

| Input | Used by | SHA-256 |
|---|---|---|
| `table3/queries.jsonl`: test spectra with the collision energies imputed as in Appendix A.5 | Table 3 MIST, ICEBERG, GLACIER | `a9a86139…a416da` |
| `table3/candidates_formula.jsonl`: the formula pool in the forward-model order | Table 3 MIST, ICEBERG, GLACIER | `b83b94ad…7fad` |
| `posteriors/mist_4m_aug_table3/`: MIST *4M-aug* posteriors | Table 3 MIST | `18d2848a…0563` (probabilities) |
| `posteriors/mist_4m_aug_table5/`: MIST *4M-aug* posteriors | Table 5 MolForge, FRIGID | `cf61afd2…b42c` (probabilities) |

Model weights go under `checkpoints/`; each `config.yaml` pins their hashes. Download
locations will be added here when the assets are published.
