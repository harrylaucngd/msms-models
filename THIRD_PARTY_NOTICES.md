# Third-party implementations

The root MIT license covers the `msms-models` interface, the evaluation code and the
implementations written here. It does not replace the licenses of the components below,
which keep their own, nor the terms of separately downloaded weights and datasets.

| Component | Source | License | In this repository |
|---|---|---|---|
| Sub-formula nearest neighbor | [serenaklm/ML_MS_analysis](https://github.com/serenaklm/ML_MS_analysis), `6817e9bd994f` | MIT; [license](src/msms_models/models/retrieval/nearest_neighbor/vendor/LICENSE.txt) | `models/retrieval/nearest_neighbor/vendor` |
| MassSpecGym baselines | [pluskal-lab/MassSpecGym](https://github.com/pluskal-lab/MassSpecGym), `444395af2c` (DeepSets), `9e6bb0f52b` (Fourier features), `e5080d6126` (Fingerprint FFN), `485deb22df` (transforms, SMILES Transformer, random generation) | MIT; [license](src/msms_models/massspecgym/vendor/LICENSE.txt) | `massspecgym/`, `models/retrieval/{random,deepsets,fingerprint_ffn}.py`, `models/de_novo/{random_generation,smiles_transformer}` |
| MIST (MistNet of the DiffMS and MassSpecGym releases) | [coleygroup/DiffMS](https://github.com/coleygroup/DiffMS), `9d9f4fd497162eec045f7db2787da30ce69a9622` (`src/mist`, from [samgoldman97/mist](https://github.com/samgoldman97/mist)) | MIT; [license](src/msms_models/encoders/mist/vendor/LICENSE.txt) | `encoders/mist/vendor` |
| ScaledMIST | [coleygroup/MIST](https://github.com/coleygroup/MIST), `4d09208` | MIT | installed package (`mist` extra) |
| DreaMS layers | [pluskal-lab/DreaMS](https://github.com/pluskal-lab/DreaMS), `02be141a2dd101e132a835ba6fea414048564a28` | MIT; [license](src/msms_models/models/retrieval/dreams_chemberta/vendor/LICENSE.txt) | `models/retrieval/dreams_chemberta/vendor` |
| DreaMS + ChemBERTa alignment | [HassounLab/SpecBridge](https://github.com/HassounLab/SpecBridge), `6c6fadf` | MIT (declared in its README) | `models/retrieval/dreams_chemberta` |
| Chemformer two-stage retrieval | [arXiv:2511.06259](https://arxiv.org/abs/2511.06259), `c6327e2`; Chemformer from [MolecularAI/Chemformer](https://github.com/MolecularAI/Chemformer) | Chemformer Apache-2.0 | `models/retrieval/chemformer_two_stage` |
| ICEBERG inference and spectral utilities | [coleygroup/ms-pred](https://github.com/coleygroup/ms-pred), `e446eebb0f83e53ede016c62522ac2dd371801de` | MIT; [license](src/msms_models/simulators/iceberg/vendor/LICENSE.txt) | `simulators/iceberg/vendor` |
| Sparse spectral scoring | [coleygroup/ms-pred](https://github.com/coleygroup/ms-pred), `708148c2a8eb0c32120644436fefd2fe90cd2214` | MIT; [license](src/msms_models/simulators/vendor/LICENSE.txt) | `simulators/vendor` |
| GLACIER | [coleygroup/ms-pred](https://github.com/coleygroup/ms-pred), `708148c` | MIT | installed package (`glacier` extra) |
| MolForge decoder | [knu-lcbc/MolForge](https://github.com/knu-lcbc/MolForge) | **CC BY-NC-SA 4.0**; [license](src/msms_models/models/de_novo/molforge/vendor/LICENSE.txt) | `models/de_novo/molforge/vendor` |
| MS-GPT SAFE decoder | [VIKI623/MS-GPT](https://github.com/VIKI623/MS-GPT), `f2c844e61f07` | Apache-2.0; [license](src/msms_models/models/de_novo/ms_gpt/vendor/LICENSE.txt) | `models/de_novo/ms_gpt/vendor` |
| FRIGID DLM and refinement | [coleygroup/FRIGID](https://github.com/coleygroup/FRIGID), `d242a2c06d671e0fb6fbc1a5b31f0e19e24b456d` | Apache-2.0; [license](src/msms_models/models/de_novo/frigid/vendor/LICENSE.txt) | `models/de_novo/frigid/vendor` (with NVIDIA notices) |
| FOAM graph optimization | [coleygroup/foam](https://github.com/coleygroup/foam), `f3ed69b11642c01be0301dd2cfcba5cb3ee8c793`; distance helper from ms-pred `00948ecfc171b3480e549c31bf765197fa0eb30d` | MIT; [license](src/msms_models/models/de_novo/foam/vendor/LICENSE.txt) | `models/de_novo/foam/vendor`, `paper/table5/foam/build_seeds.py` |
| ctMSNovelist decoder (Morgan-4096 adaptation) | Kai Zhao, Yanmin Liu, Longyang Dian, Shiwei Sun and Xuefeng Cui; adapted by Mrunali Manjrekar (`foam_baselines`, `2aed7c9c67c5ccf32adc032bc099b2a403106fd5`, `baselines/ctMSNovelist`) | MIT; [license](src/msms_models/models/de_novo/msnovelist/vendor/LICENSE.txt) | `models/de_novo/msnovelist/vendor` |

MolForge's decoder is subject to its noncommercial and share-alike terms, so the
repository as a whole must not be described as exclusively MIT.
