# Assets

Model weights, tokenizers and frozen inputs are explicit files named in each
configuration and pinned by SHA-256. A paper row expects them under `checkpoints/` and
`data/paper/`, relative to the repository root. The official MassSpecGym files are
downloaded by `msms-models download` (see [data](data.md)).

The paper's assets are published in the Hugging Face repository
[`harrylau/msms-models`](https://huggingface.co/harrylau/msms-models) with the layout below.
Download it into the repository root with `hf download harrylau/msms-models --local-dir .`.
The decoder pretraining sets of Figure 2 are in
[`harrylau/ms-gpt-18`](https://huggingface.co/datasets/harrylau/ms-gpt-18). Files distributed by their
original authors are linked instead of being copied.

## Frozen inputs (`data/paper/`)

| File | Used by | SHA-256 | Contents |
|---|---|---|---|
| `table3/queries.jsonl` | Table 3 MIST, ICEBERG, GLACIER | `a9a861398b764f3206fabb58f248ea0eb8a02e73c29dfce3a43452a925a416da` | 17,556 test spectra with collision energies imputed as in Appendix A.5 |
| `table3/candidates_formula.jsonl` | same | `b83b94ad2cbe362500e7af4c9a2b5d964f7606d27268d5ebdef9073afd0d7fad` | formula pool in the forward-model order (2,894,073 rows) |
| `table5/mist_vdiffms/subformulae.jsonl` | Table 5 vDiffMS rows | `816db026b117d94a45864243091f06cd87ea76148eb51a655768f4937266c7b0` | DiffMS subformula trees for the 17,082-spectrum cohort (40 MB) |
| `table3/pubchem_order.jsonl` | Table 3 (4) | not yet frozen | PubChem formula listings; `build.py` creates one |
| `posteriors/mist_4m_aug_table3/probabilities.npy` | Table 3 MIST | `18d2848a12861759f6d47c2b7f084361f7c31bd950cd9414980369ec451c0563` | MIST *4M-aug* posteriors (17,556 × 4,096); with `query_ids.npy` and `manifest.json` |
| `posteriors/mist_4m_aug_table5/probabilities.npy` | Table 5 MolForge, FRIGID | `cf61afd26ed94f16f8c8f0dbe0a411815597eeaf96229163c0c5d0f9ab29b42c` | MIST *4M-aug* posteriors used by the de novo rows |

Both posterior caches come from checkpoint `548c1301…659b6` (EMA weights). They were
exported separately for the retrieval and de novo experiments and are calibrated with
slightly different thresholds (0.18791085 and 0.18791015). Each row pins the cache
it used.

## Checkpoints (`checkpoints/`)

| Path | Size | SHA-256 | Row | Source |
|---|---|---|---|---|
| `mist_msg_aug.ckpt` | 289 MB | `548c1301d9093c732f4a79666e118fb87924de97576fd7c5bd3f17a104e659b6` | MIST *4M-aug* (to recompute posteriors) | [coleygroup/MIST](https://github.com/coleygroup/MIST) |
| `molforge/decoder.pth` | 556 MB | `67710e7ba8ce453e8de3b9848df482861703c7abc1d9d2e75c6f6ed984d76822` | Table 5 MolForge | [MolForge](https://github.com/knu-lcbc/MolForge) (CC BY-NC-SA 4.0) |
| `molforge/source.model` | 0.3 MB | `0b10778d86d2966e91a2ffb821c9b842b083ca4911ad92478695f6b737e204ec` | Table 5 MolForge | same |
| `molforge/target.model` | 0.2 MB | `c90c36a784b35d42fdfa9586fcb3b4d7fa35586898707328d4266a8fcb017eef` | Table 5 MolForge | same |
| `frigid/decoder.ckpt` | 2.7 GB | `b6177c2d43448380aba80ff41c01461ea34ca2ca93b213986954c5afb7f0f457` | Table 5 FRIGID | [Zenodo 19685145](https://zenodo.org/records/19685145) (`DLM.ckpt`) |
| `frigid/tokenizer.json` | 46 kB | `0db5f4dbdc7e8ff759e98483759611a426e187ee7f3f0a91edc8800abe7bf140` | Table 5 FRIGID | SAFE tokenizer paired with the DLM |
| `frigid/msg_length.joblib` | 0.3 MB | `9cab2fabfe3cae3f0efcc94f66e187494b861f47dfff59735677843a6d1e12f0` | Table 5 FRIGID | original NGBoost length predictor |
| `frigid/msg_length_features.json` | 53 B | `42c60d412a467235342bfdf44d634f3c2e30a646daf06ec7166873f437ece44e` | Table 5 FRIGID | its feature order |
| `iceberg/msg_generator.ckpt` | 42 MB | `748761b24ed50726a8d63ce68d34c37e4a8eaabfcbbd470796a3f38c35b7a253` | Table 5 FRIGID refinement | [ms-pred](https://github.com/coleygroup/ms-pred) |
| `iceberg/msg_intensity.ckpt` | 41 MB | `ec461ef26d61ce2ef3ea2269b879afcdd99a03822f9e311558ad5a3909a58ac2` | Table 5 FRIGID refinement | same |
| `iceberg/msg_ce_completed_{generator,intensity}.ckpt` | — | not located | Table 3 ICEBERG | retrained on the energy-completed split (Appendix A.5) |
| `glacier/msg_ce_completed.ckpt` | 163 MB | `b4d2612789b091906aa047e207e13f7a8d462a0b9484bac02bc65b30fc451c09` | Table 3 GLACIER | GLACIER retrained on the energy-completed split |
| `chemformer/step=1000000.ckpt` | 224 MB | `6863bdbed84e22d4f1a33f237116f6b80f74482fcbb0dd683a0ba3a9d83ea0cb` | Table 3 (2) | Chemformer pretrained BART ([MolecularAI](https://github.com/MolecularAI/Chemformer), Apache-2.0) |
| `chemformer_two_stage/inflated/pre_retrieval.pt` | 227 MB | `f87be96c6b55e576d5a79ad2978f1a7a60834609a5e95b8a914b5802c898d3fc` | Table 3 (2) inflated | authors |
| `chemformer_two_stage/inflated/generative.pt` | 321 MB | `4ba27690b888a2dc25727ee63520783cdd5aac2f5c0bd7430ca666528efb69fe` | Table 3 (2) inflated | authors |
| `chemformer_two_stage/corrected/pre_retrieval.pt` | 227 MB | `35ed098523b8041185ab5755c393168dab40357331b416c490786c9b3bedf8a7` | Table 3 (2) corrected | authors |
| `chemformer_two_stage/corrected/generative.pt` | 321 MB | `5922ba4a74132d93437cdbe1a5913f168e4cb86c04ff14f6acd34808e2221366` | Table 3 (2) corrected | authors |
| `dreams_chemberta/inflated.pt` | 714 MB | `65de203394044dff74cc675566e7d98d32a13d44e04575b06f4171ebf54ce276` | Table 3 (1) inflated | authors; optimizer state removed |
| `dreams_chemberta/corrected.pt` | 714 MB | `8d5ee7fd92d87d9199cd4cbb14314406d1c0b48bc8d7a95d5689476f3aeec398` | Table 3 (1) corrected | authors; optimizer state removed |
| `dreams/ssl_model.ckpt` | 1.4 GB | `4b73da583a4b4e4abef4bb3ab496dc12f716ed484ea0e4066ad45d6952856fef` | training (1) only | [DreaMS, Zenodo 10997887](https://zenodo.org/records/10997887) |
| `dreams_chemberta/chemberta_mlm_ik14_excluded.pt` | 528 MB | `3b65bdfdae77f57bef0a33bc3d81381fefac3095f0d0ddcd0e00834bc0c4287a` | training (1) inflated only | authors' continued ChemBERTa pretraining |
| `mist_vdiffms/mist_msg.pt` | 93 MB | `620c21febfc76a5f6a6cbacca6a4a6b8d5d380edebcfb55735ae3d3d2d806dc0` | Table 5 vDiffMS rows | [DiffMS, Zenodo 15122968](https://zenodo.org/records/15122968) |

Generated locally rather than downloaded:
- the nearest-neighbour index (`msms-models fit-nn`);
- the ChemBERTa classifiers (`chemberta_classifier/train.py`);
- MIST posteriors recomputed from a checkpoint.

The MS-GPT decoder and tokenizer come from the [MS-GPT release](https://github.com/VIKI623/MS-GPT); it is
conditioned on the MIST *4M-aug* posteriors above with the band in `configs/ms_gpt/msg-calibration.json`;
see its [guide](../src/msms_models/models/de_novo/ms_gpt/README.md).

## Hugging Face upload

Each file below goes to the given path of the Hugging Face repository. The source is
on Engaging (`$L` = `/orcd/data/ccoley/001/liuhx25`). The staging folder
`~/orcd/scratch/msms-models-hf/` already links every available file at its target
path, and its `README.md` is the model card ([`paper/huggingface_card.md`](../paper/huggingface_card.md)).
The whole tree is uploaded at once:

```bash
hf repo create harrylau/msms-models --repo-type model
hf upload harrylau/msms-models ~/orcd/scratch/msms-models-hf . --repo-type model
```

| Target path | Source on Engaging |
|---|---|
| `data/paper/table3/queries.jsonl` | `~/orcd/scratch/msms-models-parity/table3_inputs/queries.jsonl` |
| `data/paper/table3/candidates_formula.jsonl` | `~/orcd/scratch/msms-models-parity/table3_inputs/candidates_formula.jsonl` |
| `data/paper/posteriors/mist_4m_aug_table3/*` | `~/orcd/scratch/msms-models-parity/posteriors/mist_4m_aug_table3/` |
| `data/paper/posteriors/mist_4m_aug_table5/*` | `~/orcd/scratch/msms-models-parity/posteriors/mist_4m_aug_table5/` |
| `data/paper/table5/mist_vdiffms/subformulae.jsonl` | `~/orcd/scratch/msms-models-port-diffms/features/mist_vdiffms_subformulae.jsonl` |
| `checkpoints/mist_msg_aug.ckpt` | `$L/MIST-final/checkpoints/mist_msg_base.ckpt` |
| `checkpoints/molforge/{decoder.pth,source.model,target.model}` | `$L/MIST-MolForge/checkpoints/{decoder_molforge.pth,molforge_sp/combined_morgan4096_vocab_sp.model,molforge_sp/combined_smiles_vocab_sp_morgan4096.model}` |
| `checkpoints/frigid/tokenizer.json` | `$L/MSReasoner/MS-GPT/checkpoints/safe_gpt_base/tokenizer.json` |
| `checkpoints/frigid/msg_length.joblib`, `msg_length_features.json` | `$L/frigid_rebenchmark_20260918/assets/token_models/msg/best_ngboost_MSG{.joblib,_features.json}` |
| `checkpoints/iceberg/msg_{generator,intensity}.ckpt` | `$L/iceberg/iceberg_msg_model{1,2}.ckpt` |
| `checkpoints/glacier/msg_ce_completed.ckpt` | `$L/msg_wild_revision_20261003/table3_glacier_20261007/training/joint/version_0/best.ckpt` |
| `checkpoints/chemformer/step=1000000.ckpt` | `/orcd/pool/007/liuhx25/repo/GLMR/checkpoints/step=1000000.ckpt` |
| `checkpoints/chemformer_two_stage/{inflated,corrected}/{pre_retrieval,generative}.pt` | `/orcd/pool/007/liuhx25/checkpoints/GLMR/{outputs,outputs_new}/{pre_retrieval/best_model.pt,generative/best_gen_model.pt}` |
| `checkpoints/dreams_chemberta/{inflated,corrected}.pt` | `~/orcd/scratch/msms-models-port-specbridge/ckpt/{inflated,corrected}.pt` |
| `checkpoints/dreams_chemberta/chemberta_mlm_ik14_excluded.pt` | `/orcd/pool/007/liuhx25/repo/ChemBERTa-Derify/runs/derify_augmented_pubchem_13m_mlm_massspecgym_test_ik14_excluded_canonical_smiles/checkpoint_best.pt` |

Linked rather than copied, because their authors distribute them: the FRIGID
`DLM.ckpt` (Zenodo 19685145), the DreaMS SSL checkpoint (Zenodo 10997887) and the
DiffMS MIST checkpoint (Zenodo 15122968).

Before publishing:
- Check the redistribution terms: MolForge is CC BY-NC-SA 4.0, and SpecBridge and GLMR have no LICENSE file.
- These files are not yet located or created:
  - the ICEBERG pair retrained for Table 3;
  - the frozen PubChem listing.
