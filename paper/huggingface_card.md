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

Model weights and frozen inputs for the implemented entries of *MassSpecGym in the Wild:
Uncovering and Correcting Evaluation Pitfalls in AI-Driven Molecule Discovery*. The code,
configurations and per-entry instructions are in
[github.com/harrylaucngd/msms-models](https://github.com/harrylaucngd/msms-models).

Download into the root of a checkout; the paper configurations expect this layout:

```bash
bash environments/install.sh && source .venv/bin/activate
bash paper/prepare_data.sh
hf download harrylau/msms-models --local-dir . --include 'checkpoints/*' 'data/*'
bash paper/table3/mist_4m_aug/run.sh
```

| Path | Contents |
|---|---|
| `data/paper/posteriors/` | MIST fingerprint posteriors: the Table 1 encoders, MIST *4M-aug* with MIST-CF or ground-truth formulas (Table 2), and MIST *4M-aug* for Tables 3–5 and Figure 2 |
| `data/paper/table2/` | MIST-CF top-1 formulas behind the Table 2 posteriors |
| `data/paper/table3/` | Table 3 test queries with imputed collision energies, the forward-model candidate pool, and the MIST vMassSpecGym subformula trees |
| `data/paper/table5/mist_vdiffms/` | DiffMS subformula trees |
| `checkpoints/mist_msg_aug.ckpt` | MIST *4M-aug* (MassSpecGym training set + GLACIER synthetic spectra) |
| `checkpoints/mist_vmassspecgym/` | MIST vMassSpecGym ([Zenodo 11580401](https://zenodo.org/records/11580401)) |
| `checkpoints/mist_vdiffms/` | MIST released with DiffMS ([Zenodo 15122968](https://zenodo.org/records/15122968)) |
| `checkpoints/molforge/` | MolForge decoder and SentencePiece models (**CC BY-NC-SA 4.0**) |
| `checkpoints/frigid/` | FRIGID DLM ([Zenodo 19685145](https://zenodo.org/records/19685145)), SAFE tokenizer and NGBoost length predictor |
| `checkpoints/iceberg/`, `checkpoints/glacier/` | ICEBERG pair for FRIGID refinement and FOAM; ICEBERG pair and GLACIER trained on the energy-completed split for Table 3 |
| `checkpoints/chemformer/`, `checkpoints/chemformer_two_stage/` | Chemformer two-stage retrieval (inflated and corrected) |
| `checkpoints/dreams_chemberta/` | DreaMS + ChemBERTa alignment (inflated and corrected) |

The decoder pretraining sets of the paper's Figure 2 are in
[harrylau/ms-gpt-18](https://huggingface.co/datasets/harrylau/ms-gpt-18).

Each component keeps the license of its original method; see the code repository's
third-party notices. The inflated checkpoints are the paper's shortcut-learning
examples, not valid benchmark models.
