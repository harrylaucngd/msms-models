# MIST as released with DiffMS

Source: [coleygroup/DiffMS](https://github.com/coleygroup/DiffMS) at
`9d9f4fd497162eec045f7db2787da30ce69a9622`, `src/mist/`, MIT license
([LICENSE.txt](LICENSE.txt), copyright 2025 Bohde, Manjrekar, Wang, Ji and Coley).
DiffMS adapted this code from [samgoldman97/mist](https://github.com/samgoldman97/mist)
(MIT). It is the encoder that MIST-MolForge (`harrylaucngd/MIST-MolForge` @`7736e4a`)
ran for the paper's vDiffMS rows. The `mist/models` and `mist/data` code of that
MIST-MolForge revision is logically identical to this DiffMS revision on the
inference path (abstract-syntax comparison, docstrings and comments ignored).

| File | Upstream | Changes |
|---|---|---|
| `modules.py` | `src/mist/models/modules.py` | Package-local imports; `additive_bool_mask` is passed to the attention layers. |
| `spectra_encoder.py` | `src/mist/models/spectra_encoder.py` | None. |
| `form_embedders.py` | `src/mist/models/form_embedders.py` | Package-local import of `NORM_VEC`. |
| `transformer_layer.py` | `src/mist/models/transformer_layer.py` | `additive_bool_mask` selects the historical padding; the default excludes padding. |
| `chem_utils.py` | `src/mist/utils/chem_utils.py` | Only the constants and functions used by featurization and the encoder; the RDKit mass tables are omitted. |
| `featurizers.py` | `PeakFormula` in `src/mist/data/featurizers.py` | Evaluation path only: the subformula tree is passed in; augmentation, MAGMA supervision, peak capping and the unused fragment m/z vector are omitted. |

`transformer_layer.py` is DiffMS's `src/mist/models/transformer_layer.py` (logically identical
apart from the padding mask). DiffMS adds the raw boolean mask (`attn += attn_mask`, +1 on
padding); this copy adds `-inf` unless `additive_bool_mask=True`. DiffMS fixed the same line
upstream in `50b692c`.
