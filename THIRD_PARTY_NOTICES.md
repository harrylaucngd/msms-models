# Third-party implementations

The root MIT license covers the original `msms-models` interface and evaluation
code. It does not replace the licenses of the implementations listed below or
of separately downloaded weights and datasets.

| Component | Source | License | Included changes |
|---|---|---|---|
| Sub-formula nearest neighbor | [serenaklm/ML_MS_analysis](https://github.com/serenaklm/ML_MS_analysis), `6817e9bd994f` | MIT; [license](src/msms_models/models/retrieval/nearest_neighbor/vendor/LICENSE.txt) | Extracted formula annotation, sparse features and neighbor weighting into an importable inference module. |
| MolForge decoder | [knu-lcbc/MolForge](https://github.com/knu-lcbc/MolForge) | **CC BY-NC-SA 4.0**; [license](src/msms_models/models/de_novo/molforge/vendor/LICENSE.txt) | Retained the fingerprint-token transformer and beam search. Reduced configuration to inference constants and removed import-time directory creation. |
| MS-GPT SAFE decoder | [VIKI623/MS-GPT](https://github.com/VIKI623/MS-GPT), `f2c844e61f07` | Apache-2.0; [license](src/msms_models/models/de_novo/ms_gpt/vendor/LICENSE.txt) | Retained the official SAFE model and posterior-query functions; extracted candidate-ranking functions from `scripts/rank_candidates.py` without changing their ordering. Added explicit tokenizer assets. The decoder is conditioned on ScaledMIST posteriors only; MS-GPT's own MIST encoder and LoRA adapters are not included. |
| Sparse spectral scoring | [coleygroup/ms-pred](https://github.com/coleygroup/ms-pred), `708148c2a8eb0c32120644436fefd2fe90cd2214` | MIT; [license](src/msms_models/simulators/vendor/LICENSE.txt) | Extracted sparse entropy distance, binning and candidate score expansion without training or command-line dependencies. |

MolForge's included decoder is subject to its noncommercial and share-alike
terms. The repository as a whole must not be described as exclusively MIT.
The MS-GPT generation loop uses Transformers 4.49's per-block tuple KV cache in
place of the newer `DynamicCache` API so that SAFE-Mol 0.1.13 can import. A token
test compares cached greedy decoding against full-prefix model evaluation.

Installations of MIST, ms-pred, FOAM and FRIGID retain their upstream
licenses; their model weights and datasets may have separate terms.

Additional inference components:

| Component | Source | License | Included changes |
|---|---|---|---|
| FRIGID DLM and refinement | [coleygroup/FRIGID](https://github.com/coleygroup/FRIGID), based on `d242a2c06d671e0fb6fbc1a5b31f0e19e24b456d` | Apache-2.0; [license](src/msms_models/models/de_novo/frigid/vendor/LICENSE.txt) | Explicit offline tokenizer, audited shared-attention aliases, package-local imports, native refinement with an in-process spectral predictor, terminal prediction omitted when unused. Model and sampler components retain their NVIDIA notices. |
| FOAM graph optimization | [coleygroup/foam](https://github.com/coleygroup/foam), `f3ed69b11642c01be0301dd2cfcba5cb3ee8c793` | MIT; [license](src/msms_models/models/de_novo/foam/vendor/LICENSE.txt) | Native crossover, mutation and selection; package-local imports; explicit deployment seeds and entropy/SA oracle; omitted plots, target-dependent diagnostics and external logging. |
| Legacy ICEBERG inference and spectral utilities | [coleygroup/ms-pred](https://github.com/coleygroup/ms-pred), based on `e446eebb0f83e53ede016c62522ac2dd371801de` | MIT; [license](src/msms_models/simulators/iceberg/vendor/LICENSE.txt) | Isolated namespace, inference-only imports, tensor-valued DGL batch metadata, adduct and instrument vocabularies selected from the checkpoint (released weights and the ms-pred development tree of the Table 3 retrain). |

The inference adaptations are part of this repository's implementation; an
upstream revision alone does not include those adaptations. Except for the explicitly
documented local training adaptations below, training and full upstream applications
remain in their original repositories. The included FOAM
dense distance helper follows `ms-pred` revision `00948ecfc171b3480e549c31bf765197fa0eb30d`.

Paper reproduction components:

| Component | Source | License | Included changes |
|---|---|---|---|
| MIST as released with DiffMS (vDiffMS) | [coleygroup/DiffMS](https://github.com/coleygroup/DiffMS), `9d9f4fd497162eec045f7db2787da30ce69a9622` (`src/mist`, adapted from samgoldman97/mist) | MIT; [license](src/msms_models/encoders/mist/vendor/LICENSE.txt) | Retained FormulaTransformer, fingerprint growing, pos-cos embedders and the evaluation path of PeakFormula featurization/collation; package-local imports; trees passed by ID; augmentation and MAGMA removed. **Padding-mask fix** (paper Sec. 6.1.1): DiffMS adds the raw boolean mask (`attn += attn_mask`), giving padding +1 instead of -inf; the default adds the -inf mask and `attention_padding: additive-bool` restores the upstream behaviour for the diagnostic rows. |
| DreaMS layers | [pluskal-lab/DreaMS](https://github.com/pluskal-lab/DreaMS), `02be141a2dd101e132a835ba6fea414048564a28` | MIT; [license](src/msms_models/models/retrieval/dreams_chemberta/vendor/LICENSE.txt) | `layers.py`, `fourier_features.py` and `feed_forward.py` verbatim apart from a package-local import. |
| DreaMS + ChemBERTa alignment | [HassounLab/SpecBridge](https://github.com/HassounLab/SpecBridge), `6c6fadf`, with the authors' local changes | README declares MIT; the repository has no LICENSE file | Adapter, residual mapper, candidate shuffling and training reimplemented logic-identically in `models/retrieval/dreams_chemberta`; wandb, demos, AMP and resume removed. |
| MassSpecGym baselines | [pluskal-lab/MassSpecGym](https://github.com/pluskal-lab/MassSpecGym), `444395af2c` (DeepSets), `9e6bb0f52b` (Fourier features), `e5080d6126` (Fingerprint FFN), `485deb22df` (transforms, SMILES Transformer, random generation) | MIT; [license](src/msms_models/massspecgym/vendor/LICENSE.txt) | `FourierFeatures` and the random de novo generator (`models/de_novo/random_generation/vendor`) verbatim apart from removed Lightning plumbing. Transforms, networks, tokenizers and the training loop reimplemented in `massspecgym/` without Lightning, PyTorch Geometric or matchms, with identical outputs (checked against the original code). |
| Chemformer two-stage retrieval (GLMR) | the authors' reimplementation of [arXiv:2511.06259](https://arxiv.org/abs/2511.06259), `c6327e2`; Chemformer architecture from [MolecularAI/Chemformer](https://github.com/MolecularAI/Chemformer) | GLMR has no LICENSE file; Chemformer Apache-2.0 | Model, tokenizer, data and training ported with identical parameter names and calls; the Chemformer checkpoint loads without Lightning/DeepSpeed; `training.legacy_scaled_clip` keeps the older clipping used for the corrected checkpoint. |

## ctMSNovelist Morgan/MIST adaptation

`models/de_novo/msnovelist/vendor` retains the MIT-licensed ctMSNovelist decoder
and tokenizer (copyright 2024 Xuefeng Cui); see its
[license](src/msms_models/models/de_novo/msnovelist/vendor/LICENSE.txt).
Source: the contributor's `mrunalimanj/foam_baselines` checkout, revision
`2aed7c9c67c5ccf32adc032bc099b2a403106fd5`, `baselines/ctMSNovelist`,
including local configuration/training changes and the `ctMSNovelist_minimal` copy.
The source README attributes *How to Train Your Neural Network for Molecular
Structure Generation from Mass Spectra?* to Kai Zhao, Yanmin Liu, Longyang Dian,
Shiwei Sun and Xuefeng Cui.

Included changes: package-local imports, removal of unused experiment helpers,
device-aware nonpersistent buffers/temporary tensors, standalone training from the
prepared Morgan-4096 database, shared MIST input handling, and RDKit normalization
in place of the Java/SIRIUS wrapper. Architecture, vocabulary, fingerprint rounding,
parameter names and native beam scoring are retained. Training entry/data/loss code
is adapted from the same local source. Original CSI:FingerID, co-training and
MS2SMILES experiments are excluded. Model weights and data remain separate assets
and are not licensed by this notice.
