---
license: other
license_name: mixed
license_link: https://github.com/harrylaucngd/msms-models/blob/main/THIRD_PARTY_NOTICES.md
tags:
  - mass-spectrometry
  - massspecgym
  - molecule-retrieval
  - de-novo-generation
---

# msms-models: weights and frozen inputs for *MassSpecGym in the Wild*

Model weights and frozen inputs needed to reproduce the retrieval (Table 3) and
*de novo* (Table 5) rows of *MassSpecGym in the Wild: Uncovering and Correcting
Evaluation Pitfalls in AI-Driven Molecule Discovery*. The code, configurations and
per-row instructions are in [github.com/harrylaucngd/msms-models](https://github.com/harrylaucngd/msms-models).

Download into the root of a checkout; the paper configurations expect this layout
and check every file's SHA-256 (listed in `docs/assets.md` of the code repository):

```bash
hf download harrylau/msms-models --local-dir .
bash paper/prepare_data.sh
bash paper/table3/mist_4m_aug/run.sh
```

| Path | Contents |
|---|---|
| `data/paper/table3/` | Table 3 test queries with imputed collision energies, and the forward-model candidate pool |
| `data/paper/posteriors/` | MIST *4M-aug* fingerprint posteriors used by the Table 3 and Table 5 rows |
| `data/paper/table5/mist_vdiffms/` | DiffMS subformula trees for the vDiffMS diagnostic rows |
| `checkpoints/mist_msg_aug.ckpt` | MIST *4M-aug* (MassSpecGym training set + GLACIER synthetic spectra) |
| `checkpoints/molforge/` | MolForge decoder and SentencePiece models (**CC BY-NC-SA 4.0**) |
| `checkpoints/frigid/` | SAFE tokenizer and NGBoost length predictor paired with the FRIGID DLM |
| `checkpoints/iceberg/`, `checkpoints/glacier/` | ICEBERG pair used for FRIGID refinement; GLACIER retrained on the energy-completed split |
| `checkpoints/chemformer/`, `checkpoints/chemformer_two_stage/` | Chemformer two-stage retrieval (inflated and corrected) |
| `checkpoints/dreams_chemberta/` | DreaMS + ChemBERTa alignment (inflated and corrected) |

Downloaded separately from their original sources:
- FRIGID `DLM.ckpt`: [Zenodo 19685145](https://zenodo.org/records/19685145).
- The DreaMS SSL checkpoint: [Zenodo 10997887](https://zenodo.org/records/10997887).
- The DiffMS MIST: [Zenodo 15122968](https://zenodo.org/records/15122968).

The decoder pretraining sets of the paper's Figure 2 are in
[harrylau/ms-gpt-18](https://huggingface.co/datasets/harrylau/ms-gpt-18).

Each component keeps the license of its original method; see the code repository's
third-party notices. Inflated checkpoints reproduce the paper's shortcut-learning
diagnostics and are not valid benchmark models.
