# Reproducing *MassSpecGym in the Wild*

Every method of Tables 2, 3 and 5 that can be run from released code has one folder here:
the authors' own rows and the MassSpecGym baselines, reimplemented. Each folder holds
the exact configurations, a `run.sh` that predicts and evaluates, and a README stating
where the method comes from and what it needs. Method source lives once, in
[`src/msms_models`](../src/msms_models/); the folders only bind it to the paper's inputs
and settings.

```text
paper/
  prepare_data.sh                  official MassSpecGym inputs
  table2/<row>/                    README.md, config*.yaml, run.sh
  table3/<row>/
  table5/<row>/
```

## Table 2 · Retrieval on mass-based candidates

| Row | Folder | Origin |
|---|---|---|
| Random, Random w/ ground-truth formula | [random](table2/random/) | MassSpecGym baseline |

## Table 3 · Retrieval with known formula

| Row | Folder | Origin |
|---|---|---|
| Random | [random](table3/random/) | MassSpecGym baseline |
| DeepSets, DeepSets + Fourier features | [deepsets](table3/deepsets/) | MassSpecGym baseline, reimplementation |
| Fingerprint FFN | [fingerprint_ffn](table3/fingerprint_ffn/) | MassSpecGym baseline, reimplementation |
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
| SMILES Transformer, SELFIES Transformer | [smiles_transformer](table5/smiles_transformer/) | MassSpecGym baseline, reimplementation |
| Random generation | [random_generation](table5/random_generation/) | MassSpecGym baseline |
| MIST + MolForge† (corrected) | [mist_molforge](table5/mist_molforge/) | reimplementation |
| MIST + MolForge† (vDiffMS, batch size 24 / 1; Table 4) | [mist_vdiffms_molforge](table5/mist_vdiffms_molforge/) | diagnostic |
| FRIGID-base | [frigid_base](table5/frigid_base/) | author rerun |
| FRIGID (10 rounds of refinement) | [frigid](table5/frigid/) | author rerun |

MassSpecGym released no checkpoints for its baselines, so their `run.sh` scripts train
them with MassSpecGym's own settings; the retrained models reproduce the printed numbers
within their uncertainty, not to the digit. The remaining rows were copied from the
original papers or the [MassSpecGym leaderboard](https://github.com/pluskal-lab/MassSpecGym)
and are not reproduced here: Table 2's MIST rows (they need MIST-CF formulas), Table 3's
MIST vMassSpecGym (its released checkpoint, Zenodo 11580401, gives Hit@1 11.04% rather
than 9.57%, so it is not the model behind the row), JESTR, MVP and FLARE, and Table 5's
other external rows, including
FRIGID with MIST vFRIGID. MIST + MSNovelist and FOAM are implemented as library models
(see the [main README](../README.md#models)), without folders.

## Running a row

```bash
bash environments/install.sh && source .venv/bin/activate   # see docs/install.md
bash paper/prepare_data.sh
bash paper/table3/mist_4m_aug/run.sh
```

Every `run.sh` is run from the repository root and writes `metrics.json` next to the
predictions.

## Evaluation protocols

`msms-models evaluate --protocol table2|table3|table5` fixes the MassSpecGym metrics
(`table2` is `table3` on the mass-based candidates):
- **Identity:** InChIKey-14.
- **Similarity:** Morgan radius-2/2048 Tanimoto without chirality.
- **MCES:** myopic MCES with threshold 15 and the stronger bound. Invalid or missing predictions count as 100.
- **Intervals:** 99.9% BCa over 20,000 spectrum-level resamples.

See [`evaluation/protocols.py`](../src/msms_models/evaluation/protocols.py).

## Inputs and assets

`prepare_data.sh` prepares the official MassSpecGym TSV and the formula and mass
candidate pools.
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
