# Assets

Model weights, tokenizers and frozen inputs are explicit files named in each
configuration. A paper row expects them under `checkpoints/` and
`data/paper/`, relative to the repository root. The official MassSpecGym files are
downloaded by `msms-models download` (see [data](data.md)).

The paper's assets are published in the Hugging Face repository
[`harrylau/msms-models`](https://huggingface.co/harrylau/msms-models) with the layout below.
Download it into the repository root with `hf download harrylau/msms-models --local-dir .`.
The decoder pretraining sets of Figure 2 are in
[`harrylau/ms-gpt-18`](https://huggingface.co/datasets/harrylau/ms-gpt-18). Files distributed by their
original authors are linked instead of being copied: the FRIGID `DLM.ckpt`, the DreaMS SSL
checkpoint and the DiffMS MIST checkpoint come from the Zenodo records in the table below.

## Frozen inputs (`data/paper/`)

| File | Used by | Contents |
|---|---|---|
| `table3/queries.jsonl` | Table 3 MIST, ICEBERG, GLACIER | 17,556 test spectra with collision energies imputed as in Appendix A.5 |
| `table3/candidates_formula.jsonl` | same | formula pool in the forward-model order (2,894,073 rows) |
| `table5/mist_vdiffms/subformulae.jsonl` | Table 5 vDiffMS rows | DiffMS subformula trees for the 17,082-spectrum cohort (40 MB) |
| `posteriors/mist_4m_aug_table3/probabilities.npy` | Table 3 MIST | MIST *4M-aug* posteriors (17,556 × 4,096); with `query_ids.npy` |
| `posteriors/mist_4m_aug_table5/probabilities.npy` | Table 5 MolForge, FRIGID | MIST *4M-aug* posteriors used by the de novo rows |

Both posterior caches come from the MIST *4M-aug* checkpoint (EMA weights). They were
exported separately for the retrieval and de novo experiments and are calibrated with
slightly different thresholds (0.18791085 and 0.18791015). Each row's configuration
names the cache it used.

## Checkpoints (`checkpoints/`)

| Path | Size | Row | Source |
|---|---|---|---|
| `mist_msg_aug.ckpt` | 289 MB | MIST *4M-aug* (to recompute posteriors) | [coleygroup/MIST](https://github.com/coleygroup/MIST) |
| `molforge/decoder.pth` | 556 MB | Table 5 MolForge | [MolForge](https://github.com/knu-lcbc/MolForge) (CC BY-NC-SA 4.0) |
| `molforge/source.model` | 0.3 MB | Table 5 MolForge | same |
| `molforge/target.model` | 0.2 MB | Table 5 MolForge | same |
| `frigid/decoder.ckpt` | 2.7 GB | Table 5 FRIGID | [Zenodo 19685145](https://zenodo.org/records/19685145) (`DLM.ckpt`) |
| `frigid/tokenizer.json` | 46 kB | Table 5 FRIGID | SAFE tokenizer paired with the DLM |
| `frigid/msg_length.joblib` | 0.3 MB | Table 5 FRIGID | original NGBoost length predictor |
| `frigid/msg_length_features.json` | 53 B | Table 5 FRIGID | its feature order |
| `iceberg/msg_generator.ckpt` | 42 MB | Table 5 FRIGID refinement | [ms-pred](https://github.com/coleygroup/ms-pred) |
| `iceberg/msg_intensity.ckpt` | 41 MB | Table 5 FRIGID refinement | same |
| `iceberg/msg_ce_completed_{generator,intensity}.ckpt` | 42 MB, 41 MB | Table 3 ICEBERG | retrained on the energy-completed split (Appendix A.5) |
| `glacier/msg_ce_completed.ckpt` | 163 MB | Table 3 GLACIER | GLACIER retrained on the energy-completed split |
| `chemformer/step=1000000.ckpt` | 224 MB | Table 3 (2) | Chemformer pretrained BART ([MolecularAI](https://github.com/MolecularAI/Chemformer), Apache-2.0) |
| `chemformer_two_stage/inflated/pre_retrieval.pt` | 227 MB | Table 3 (2) inflated | authors |
| `chemformer_two_stage/inflated/generative.pt` | 321 MB | Table 3 (2) inflated | authors |
| `chemformer_two_stage/corrected/pre_retrieval.pt` | 227 MB | Table 3 (2) corrected | authors |
| `chemformer_two_stage/corrected/generative.pt` | 321 MB | Table 3 (2) corrected | authors |
| `dreams_chemberta/inflated.pt` | 714 MB | Table 3 (1) inflated | authors; optimizer state removed |
| `dreams_chemberta/corrected.pt` | 714 MB | Table 3 (1) corrected | authors; optimizer state removed |
| `dreams/ssl_model.ckpt` | 1.4 GB | training (1) only | [DreaMS, Zenodo 10997887](https://zenodo.org/records/10997887) |
| `dreams_chemberta/chemberta_mlm_ik14_excluded.pt` | 528 MB | training (1) inflated only | authors' continued ChemBERTa pretraining |
| `mist_vdiffms/mist_msg.pt` | 93 MB | Table 5 vDiffMS rows | [DiffMS, Zenodo 15122968](https://zenodo.org/records/15122968) |

Generated locally rather than downloaded:
- the PubChem listing `data/paper/table3/pubchem_order.jsonl`, by
  `python -m msms_models.models.retrieval.pubchem_order.build` (needs internet access);
- the nearest-neighbour index (`msms-models fit-nn`);
- the ChemBERTa classifiers (`chemberta_classifier/train.py`);
- MIST posteriors recomputed from a checkpoint.

The MS-GPT decoder and tokenizer come from the [MS-GPT release](https://github.com/VIKI623/MS-GPT); it is
conditioned on the MIST *4M-aug* posteriors above with the band in `configs/ms_gpt/msg-calibration.json`;
see its [guide](../src/msms_models/models/de_novo/ms_gpt/README.md).

## Not distributed

The configurations under `configs/` also name files that are not part of the Hugging Face
release. They come from the upstream projects:
- MIST: `checkpoints/mist_msg_noaug.ckpt`, `checkpoints/mist_canopus_aug.ckpt` and the
  feature caches `data/compact/msg/test` and `data/compact/canopus/test`;
- FRIGID: `checkpoints/frigid/canopus_length.joblib` and `canopus_length_features.json`;
- ms-pred: `checkpoints/iceberg/canopus_generator.ckpt` and `canopus_intensity.ckpt`.

The FOAM seed files (`data/msg/foam_seeds.jsonl`, `data/canopus/foam_seeds.jsonl`) are
supplied by the user; see the [FOAM guide](../src/msms_models/models/de_novo/foam/README.md).

## MSNovelist (contributor assets)

The MIST + MSNovelist checkpoint and the DiffMS-era MIST posteriors it was run on are not
yet in the Hugging Face release. They come from the contributor's original baseline:

| Path | Contents |
|---|---|
| `checkpoints/msnovelist/w-12-0.034-0.019.pth` | the adapted Morgan-4096 decoder (17 MB) |
| `data/posteriors/mist_diffms_msg/` | `probabilities.npy` (17,556 × 4,096) and `query_ids.npy` |
| `data/posteriors/mist_diffms_canopus/` | the same for the 818 CANOPUS test spectra |

Training also needs the contributor's prepared database (`org_moses_0824.db`, 13 GB).
