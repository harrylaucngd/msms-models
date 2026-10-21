# Third-party implementations

The root MIT license covers the original `msms-models` interface and evaluation
code. It does not replace the licenses of the implementations listed below or
of separately downloaded weights and datasets.

| Component | Source | License | Included changes |
|---|---|---|---|
| Sub-formula nearest neighbor | [serenaklm/ML_MS_analysis](https://github.com/serenaklm/ML_MS_analysis), `6817e9bd994f` | MIT; [license](LICENSES/ML_MS_analysis-MIT.txt) | Extracted formula annotation, sparse features and neighbor weighting into an importable inference module. |
| MolForge decoder | [knu-lcbc/MolForge](https://github.com/knu-lcbc/MolForge) | **CC BY-NC-SA 4.0**; [license](src/msms_models/methods/molforge/vendor/LICENSE.txt) | Retained the fingerprint-token transformer and beam search. Reduced configuration to inference constants and removed import-time directory creation. |
| MS-GPT SAFE decoder | [VIKI623/MS-GPT](https://github.com/VIKI623/MS-GPT), `f2c844e61f07` | Apache-2.0; [license](src/msms_models/methods/ms_gpt/vendor/LICENSE.txt) | Retained the official SAFE model and posterior-query functions; extracted candidate-ranking functions from `scripts/rank_candidates.py` without changing their ordering. Added explicit tokenizer assets and validated inference-only loading of posterior-aligned adapters. |
| Historical benchmark-paired MIST | [VIKI623/MS-GPT](https://github.com/VIKI623/MS-GPT), `f2c844e61f079051baf3014c33df5ba2f0524363`, derived from MIST/FRIGID | Apache-2.0; [license](src/msms_models/methods/mist/vendor/LICENSE.txt) | Retained formula attention, fingerprint growing and spectrum featurization. The wrapper reads only spectrum columns, aligns IDs and audits checkpoint keys; it preserves the official MS-GPT posterior path and its bypassed intermediate projection. **Fixed the padding-mask bug** (paper Sec. 6.1.1): upstream adds the raw boolean mask (`attn += attn_mask`), giving padding +1 instead of -inf, so posteriors depend on batch composition. The default now adds the -inf mask; `attention_padding: additive-bool` restores the upstream behaviour for diagnostics. Batched outputs therefore differ from upstream MS-GPT. |
| Sparse spectral scoring | [coleygroup/ms-pred](https://github.com/coleygroup/ms-pred), `708148c2a8eb0c32120644436fefd2fe90cd2214` | MIT; [license](LICENSES/ms-pred-MIT.txt) | Extracted sparse entropy distance, binning and candidate score expansion without training or command-line dependencies. |

MolForge's included decoder is subject to its noncommercial and share-alike
terms. The repository as a whole must not be described as exclusively MIT.
The MS-GPT generation loop uses Transformers 4.49's per-block tuple KV cache in
place of the newer `DynamicCache` API so that SAFE-Mol 0.1.13 can import. A token
test compares cached greedy decoding against full-prefix model evaluation.
The adapter loader maps the mirror's `task_type: CUSTOM` metadata to PEFT's
generic `task_type=None` without changing adapter tensors.

Installations of MIST, ms-pred, FOAM and FRIGID retain their upstream
licenses; their model weights and datasets may have separate terms.

Additional inference components:

| Component | Source | License | Included changes |
|---|---|---|---|
| FRIGID DLM and refinement | [coleygroup/FRIGID](https://github.com/coleygroup/FRIGID), based on `d242a2c06d671e0fb6fbc1a5b31f0e19e24b456d` | Apache-2.0; [license](src/msms_models/methods/frigid/vendor/LICENSE.txt) | Explicit offline tokenizer, audited shared-attention aliases, package-local imports, native refinement with an in-process spectral predictor, terminal prediction omitted when unused. Model and sampler components retain their NVIDIA notices. |
| FOAM graph optimization | [coleygroup/foam](https://github.com/coleygroup/foam), `f3ed69b11642c01be0301dd2cfcba5cb3ee8c793` | MIT; [license](src/msms_models/methods/foam/vendor/LICENSE.txt) | Native crossover, mutation and selection; package-local imports; explicit deployment seeds and entropy/SA oracle; omitted plots, target-dependent diagnostics and external logging. |
| Legacy ICEBERG inference and spectral utilities | [coleygroup/ms-pred](https://github.com/coleygroup/ms-pred), based on `e446eebb0f83e53ede016c62522ac2dd371801de` | MIT; [license](src/msms_models/methods/iceberg/vendor/LICENSE.txt) | Isolated namespace, inference-only imports, tensor-valued DGL batch metadata, explicit released instrument vocabularies and strict weight loading. |

The inference adaptations are part of this repository's implementation; an
upstream revision alone does not include those adaptations. Training and full
upstream applications remain in their original repositories. The included FOAM
dense distance helper follows `ms-pred` revision `00948ecfc171b3480e549c31bf765197fa0eb30d`.
