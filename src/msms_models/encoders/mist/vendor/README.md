# MistNet (MIST as released with DiffMS)

The MistNet spectrum encoder used by [`diffms.py`](../diffms.py): a formula transformer over
the subformula tree of a spectrum, followed by fingerprint growing to 4,096 bits. It loads
both the checkpoint released with DiffMS and MIST vMassSpecGym.

| File | Source in DiffMS | Changes |
|---|---|---|
| `modules.py` | `src/mist/models/modules.py` | Package-local imports; `additive_bool_mask` is passed to the attention layers. |
| `spectra_encoder.py` | `src/mist/models/spectra_encoder.py` | None. |
| `form_embedders.py` | `src/mist/models/form_embedders.py` | Package-local import of `NORM_VEC`. |
| `transformer_layer.py` | `src/mist/models/transformer_layer.py` | Padding receives `-inf` in the attention scores; `additive_bool_mask=True` adds the boolean padding mask instead (+1 on padding, paper §6.1.1). |
| `chem_utils.py` | `src/mist/utils/chem_utils.py` | Only the constants and functions used by featurization and the encoder. |
| `featurizers.py` | `PeakFormula` in `src/mist/data/featurizers.py` | Inference path only: the subformula tree is passed in; augmentation, MAGMA supervision, peak capping and the unused fragment m/z vector are omitted. |

## Source

[coleygroup/DiffMS](https://github.com/coleygroup/DiffMS) at
`9d9f4fd497162eec045f7db2787da30ce69a9622`, `src/mist/`, MIT license
([LICENSE.txt](LICENSE.txt), copyright 2025 Bohde, Manjrekar, Wang, Ji and Coley).
DiffMS adapted this code from [samgoldman97/mist](https://github.com/samgoldman97/mist)
(MIT). See also the [third-party notices](../../../../../THIRD_PARTY_NOTICES.md).
