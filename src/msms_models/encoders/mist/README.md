# MIST

MIST predicts a 4,096-bit Morgan fingerprint posterior from a spectrum. `get_posteriors` in
[`__init__.py`](__init__.py) supplies these posteriors to every MIST-conditioned model: it
reads a posterior cache aligned by spectrum ID, or encodes the spectra with ScaledMIST
([`scaled.py`](scaled.py), the architecture of the pinned `coleygroup/MIST` package), or with
the MistNet of the DiffMS and MassSpecGym releases ([`diffms.py`](diffms.py) and
[`vendor/`](vendor/)). Posteriors computed during a run are written to `<output>/posterior`
in the cache format. The `mist` retrieval model
([`models/retrieval/mist.py`](../../models/retrieval/mist.py)) ranks candidates by their
similarity to the posterior; MolForge, MS-GPT, FRIGID and MSNovelist decode molecules from it.

## Assets

| Configuration key | Input |
|---|---|
| `posterior`, `posterior_ids` | A posterior cache (`probabilities.npy`) and its spectrum IDs (`query_ids.npy`); see the [data guide](../../../../docs/data.md#mist-posteriors). |
| `mist_checkpoint` | ScaledMIST EMA checkpoint (with `compact_cache`), or MistNet weights (with `diffms_features`). |
| `compact_cache` | ScaledMIST spectrum features for the dataset and split. |
| `diffms_features` | DiffMS subformula trees: a directory of `<id>.json` files or a JSONL file of `{"id", "subformulae"}` records. |
| `candidates` | ID-keyed candidate pool (`mist` retrieval). |

The encoder is chosen by the assets: `posterior` reads the cache, `diffms_features` selects
MistNet, and otherwise ScaledMIST is used. MistNet encodes the spectra in query order,
`encoder_batch_size` consecutive spectra per forward pass, each batch right-padded to its
longest spectrum.

**Parameters.**

| Key | Default | Meaning |
|---|---|---|
| `mist_architecture` | `diffms` | MistNet only: `diffms` is the checkpoint released with DiffMS (`mist_msg.pt`); `massspecgym` is MIST vMassSpecGym (Zenodo 11580401) |
| `attention_padding` | `masked` | MistNet only: `masked` excludes padding from attention; `additive-bool` adds the boolean padding mask to the attention scores (paper §6.1.1), so posteriors depend on the batch |
| `encoder_batch_size` | 64 (ScaledMIST), 32 (MistNet) | spectra per encoder forward pass |
| `fingerprint_representation` | `raw` | `mist` retrieval: `binary` thresholds the posterior before ranking; `raw` ranks by the probabilities |
| `fingerprint_threshold` | none | threshold for `binary`; calibrated for the encoder in use |
| `similarity` | `tanimoto` | `mist` retrieval: continuous `tanimoto` or `cosine` against each candidate's Morgan radius-2/4096 fingerprint |
| `candidate_ties` | `stable` | `stable` keeps the candidate order on ties; `legacy-quicksort` orders candidates by a reversed NumPy 1.24 argsort |

## Run

From the repository root, after `bash paper/prepare_data.sh`, each paper row that uses MIST
has its own script:

| Row | Script | Encoder |
|---|---|---|
| Table 1, GLACIER leakage settings 1–5 | `bash paper/table1/glacier_leakage/run.sh` | posterior caches |
| Table 2, MIST | `bash paper/table2/mist/run.sh` | MIST *4M-aug* posterior caches |
| Table 3, MIST *4M-aug* | `bash paper/table3/mist_4m_aug/run.sh` | MIST *4M-aug* posterior cache |
| Table 3, MIST vMassSpecGym | `bash paper/table3/mist_vmassspecgym/run.sh` | MistNet, `massspecgym` |
| Table 5, MIST + MolForge | `bash paper/table5/mist_molforge/run.sh` | MIST *4M-aug* posterior cache |
| Table 5, MIST vDiffMS + MolForge | `bash paper/table5/mist_vdiffms_molforge/run.sh` | MistNet, `diffms` |
| Table 5, MIST + MSNovelist | `bash paper/table5/mist_msnovelist/run.sh` | MistNet, `diffms` |

Checkpoints and posterior caches are listed in the [asset inventory](../../../../docs/assets.md).

## Source

ScaledMIST comes from [coleygroup/MIST](https://github.com/coleygroup/MIST) at
`4d09208fceaa9a34579493a70095416405b76563`, installed as a dependency. MistNet is the MIST
of [coleygroup/DiffMS](https://github.com/coleygroup/DiffMS) at `9d9f4fd`, adapted from
[samgoldman97/mist](https://github.com/samgoldman97/mist), MIT; see [`vendor/`](vendor/README.md)
and the [third-party notices](../../../../THIRD_PARTY_NOTICES.md).
