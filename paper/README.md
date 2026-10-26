# Reproducing *MassSpecGym in the Wild*

Every method the authors ran for Tables 3 and 5 has one folder here. Each folder holds
the exact configurations, a `run.sh` that predicts and evaluates, and a README stating
where the method comes from and what it needs. Method source lives once, in
[`src/msms_models`](../src/msms_models/); the folders only bind it to the paper's inputs
and settings.

```text
paper/
  prepare_data.sh                  official MassSpecGym inputs
  table3/<row>/                    README.md, config*.yaml, run.sh
  table5/<row>/
```

## Table 3 · Retrieval with known formula

| Row | Folder | Origin |
|---|---|---|
| Nearest Neighbor† | [nearest_neighbor](table3/nearest_neighbor/) | reimplementation |
| MIST *4M-aug* | [mist_4m_aug](table3/mist_4m_aug/) | author rerun |
| ICEBERG | [iceberg](table3/iceberg/) | author rerun |
| GLACIER | [glacier](table3/glacier/) | author rerun |
| (1) DreaMS + ChemBERTa, inflated / corrected | [dreams_chemberta](table3/dreams_chemberta/) | author experiment |
| (2) Chemformer two-stage, inflated / corrected | [chemformer_two_stage](table3/chemformer_two_stage/) | author experiment |
| (3) ChemBERTa SMILES classifier, inflated / corrected | [chemberta_classifier](table3/chemberta_classifier/) | author experiment |
| (4) PubChem default ranking | [pubchem_order](table3/pubchem_order/) | spectrum-blind baseline |
| (5) Ranking by number of chiral atoms | [chiral_atoms](table3/chiral_atoms/) | spectrum-blind baseline |

## Table 5 · De novo generation with known formula

| Row | Folder | Origin |
|---|---|---|
| MIST + MolForge† (corrected) | [mist_molforge](table5/mist_molforge/) | reimplementation |
| MIST + MolForge† (vDiffMS, batch size 24 / 1; Table 4) | [mist_vdiffms_molforge](table5/mist_vdiffms_molforge/) | diagnostic |
| FRIGID-base | [frigid_base](table5/frigid_base/) | author rerun |
| FRIGID (10 rounds of refinement) | [frigid](table5/frigid/) | author rerun |

The remaining rows of both tables were copied from the original papers or the
[MassSpecGym leaderboard](https://github.com/pluskal-lab/MassSpecGym) and are not
implemented here. That includes FRIGID with MIST vFRIGID, which reports the
FRIGID authors' number.

## Running a row

```bash
bash environments/install.sh && source .venv/bin/activate   # see docs/install.md
bash paper/prepare_data.sh
bash paper/table3/mist_4m_aug/run.sh
```

Every `run.sh` is run from the repository root and writes `metrics.json` next to the
predictions.

## Evaluation protocols

`msms-models evaluate --protocol table3|table5` fixes the MassSpecGym metrics:
- **Identity:** InChIKey-14.
- **Similarity:** Morgan radius-2/2048 Tanimoto without chirality.
- **MCES:** myopic MCES with threshold 15 and the stronger bound. Invalid or missing predictions count as 100.
- **Intervals:** 99.9% BCa over 20,000 spectrum-level resamples.

See [`evaluation/protocols.py`](../src/msms_models/evaluation/protocols.py).

## Inputs and assets

`prepare_data.sh` prepares the official MassSpecGym TSV and formula candidate pool.
Some rows also need frozen inputs that cannot be rebuilt from the official release.
They are placed under `data/paper/`:

| Input | Used by |
|---|---|
| `table3/queries.jsonl`: test spectra with the collision energies imputed as in Appendix A.5 | Table 3 MIST, ICEBERG, GLACIER |
| `table3/candidates_formula.jsonl`: the formula pool in the forward-model order | Table 3 MIST, ICEBERG, GLACIER |
| `posteriors/mist_4m_aug_table3/`: MIST *4M-aug* posteriors | Table 3 MIST |
| `posteriors/mist_4m_aug_table5/`: MIST *4M-aug* posteriors | Table 5 MolForge, FRIGID |
| `table5/mist_vdiffms/subformulae.jsonl`: DiffMS subformula trees | Table 5 vDiffMS rows |

Model weights go under `checkpoints/`. Both are published in [`harrylau/msms-models`](https://huggingface.co/harrylau/msms-models)
(`hf download harrylau/msms-models --local-dir .`); see [assets](../docs/assets.md).
The PubChem listing of row (4), `table3/pubchem_order.jsonl`, is not distributed: its
`run.sh` generates it locally with `pubchem_order/build.py`, which needs internet access.
The Figure 2 decoder pretraining sets are in
[`harrylau/ms-gpt-18`](https://huggingface.co/datasets/harrylau/ms-gpt-18).
