# Assets

Model weights, tokenizers and frozen inputs are explicit files named in each
configuration. An entry expects them under `checkpoints/` and `data/paper/`, relative to
the repository root. The official MassSpecGym files are downloaded by
`paper/prepare_data.sh` (see [data](data.md)); everything below is in the Hugging Face
repository [`harrylau/msms-models`](https://huggingface.co/harrylau/msms-models):

```bash
hf download harrylau/msms-models --local-dir . --include 'checkpoints/*' 'data/*'
```

The decoder pretraining sets of Figure 2 are in
[`harrylau/ms-gpt-18`](https://huggingface.co/datasets/harrylau/ms-gpt-18).

## Frozen inputs (`data/paper/`)

| File | Used by | Contents |
|---|---|---|
| `table3/queries.jsonl` | Table 3 MIST *4M-aug*, ICEBERG, GLACIER; Table 1 setting 4 | 17,556 test spectra with collision energies imputed as in Appendix A.5 |
| `table3/candidates_formula.jsonl` | same | formula pool in the forward-model order (2,894,073 rows) |
| `table3/mist_vmassspecgym/subformulae.jsonl` | Table 3 MIST vMassSpecGym | subformula trees of the MassSpecGym MIST data ([Zenodo 11580401](https://zenodo.org/records/11580401)) for the 17,556 test spectra |
| `table5/mist_vdiffms/subformulae.jsonl` | Tables 4–5 vDiffMS rows, MIST + MSNovelist | DiffMS subformula trees for the 17,082 spectra of the DiffMS evaluation |
| `table2/mist_cf_formulas.jsonl` | Table 2 MIST | the MIST-CF top-1 formula of each spectrum, per training split |
| `posteriors/mist_table1_setting{1,2,3}/` | Table 1 settings 1–3 | MIST posteriors of the GLACIER-augmented encoders (17,556 × 4,096) |
| `posteriors/mist_noaug_table1/` | Table 1 setting 5 | MIST trained without augmentation |
| `posteriors/mist_4m_aug_table2_{mistcf_standard,mistcf_testonly,mistcf_traintest,ground_truth}/` | Table 2 MIST | MIST *4M-aug* posteriors from MIST-CF or ground-truth formulas, for the spectra with a formula |
| `posteriors/mist_4m_aug_table3/` | Table 3 MIST *4M-aug*; Table 1 setting 4 | MIST *4M-aug* posteriors |
| `posteriors/mist_4m_aug_table5/` | Tables 4–5 MolForge, FRIGID; Figure 2 | MIST *4M-aug* posteriors of the de novo experiments |

Each posterior directory holds `probabilities.npy` and `query_ids.npy`. The calibrated
threshold of each is in the configuration that reads it.

## Checkpoints (`checkpoints/`)

| Path | Size | Entry | Source |
|---|---|---|---|
| `mist_msg_aug.ckpt` | 289 MB | MIST *4M-aug* | [coleygroup/MIST](https://github.com/coleygroup/MIST) |
| `mist_vmassspecgym/mist_fp.ckpt` | 103 MB | Table 3 MIST vMassSpecGym | [MassSpecGym, Zenodo 11580401](https://zenodo.org/records/11580401) |
| `mist_vdiffms/mist_msg.pt` | 93 MB | Tables 4–5 vDiffMS rows, MIST + MSNovelist | [DiffMS, Zenodo 15122968](https://zenodo.org/records/15122968) (`encoder_msg.pt`) |
| `molforge/decoder.pth` | 556 MB | Tables 4–5 MolForge | [MolForge](https://github.com/knu-lcbc/MolForge) (CC BY-NC-SA 4.0) |
| `molforge/source.model`, `molforge/target.model` | 0.3 MB, 0.2 MB | Tables 4–5 MolForge | same |
| `frigid/decoder.ckpt` | 2.7 GB | Table 5 FRIGID | [FRIGID, Zenodo 19685145](https://zenodo.org/records/19685145) (`DLM.ckpt`) |
| `frigid/tokenizer.json` | 46 kB | Table 5 FRIGID | SAFE tokenizer paired with the DLM |
| `frigid/msg_length.joblib`, `frigid/msg_length_features.json` | 0.3 MB, 53 B | Table 5 FRIGID | NGBoost length predictor and its feature order |
| `iceberg/msg_generator.ckpt`, `iceberg/msg_intensity.ckpt` | 42 MB, 41 MB | Table 5 FRIGID refinement, FOAM | [ms-pred](https://github.com/coleygroup/ms-pred) |
| `iceberg/msg_ce_completed_{generator,intensity}.ckpt` | 42 MB, 41 MB | Table 3 ICEBERG | trained on the energy-completed split (Appendix A.5) |
| `glacier/msg_ce_completed.ckpt` | 163 MB | Table 3 GLACIER | trained on the energy-completed split |
| `chemformer/step=1000000.ckpt` | 224 MB | Table 3 (2) | Chemformer pretrained BART ([MolecularAI](https://github.com/MolecularAI/Chemformer), Apache-2.0) |
| `chemformer_two_stage/{inflated,corrected}/{pre_retrieval,generative}.pt` | 227 MB, 321 MB each | Table 3 (2) | authors |
| `dreams_chemberta/{inflated,corrected}.pt` | 714 MB each | Table 3 (1) | authors |
| `dreams_chemberta/chemberta_mlm_ik14_excluded.pt` | 528 MB | training of (1) inflated | authors' continued ChemBERTa pretraining |

## Built by an entry

Some entries build their models or inputs themselves, from the files above and the
official release:
- `run.sh` trains or fits when its output is missing: the nearest-neighbour index
  (`data/indices/nearest-neighbor-msg`), DeepSets (`checkpoints/deepsets/{plain,fourier}/`),
  Fingerprint FFN (`checkpoints/fingerprint_ffn/`), the ChemBERTa classifiers
  (`checkpoints/chemberta_classifier/`), the SMILES and SELFIES Transformers
  (`checkpoints/smiles_transformer/{smiles,selfies}/`), the random generator's element
  statistics (`checkpoints/random_generation/element_stats.pkl`), the PubChem listing of
  (4) (`data/paper/table3/pubchem_order.jsonl`, needs internet access), and the cohort query
  files of Table 2 and the DiffMS rows.
- Training the DreaMS + ChemBERTa models of (1) starts from the DreaMS SSL checkpoint
  (`checkpoints/dreams/ssl_model.ckpt`, [Zenodo 10997887](https://zenodo.org/records/10997887)).
- The MS-GPT decoder and tokenizer of Figure 2 are downloaded by its `run.sh` from the
  [author-authorized mirrors](https://github.com/VIKI623/MS-GPT/issues/1).
- MIST + MSNovelist trains its decoder (`checkpoints/msnovelist/decoder.pth`) on DiffMS's
  decoder pretraining molecules; see [its README](../paper/table5/mist_msnovelist/README.md).
- FOAM builds its seeds (`data/paper/table5/foam/seeds.jsonl`) from DiffMS predictions and
  PubChem; see [its README](../paper/table5/foam/README.md).
