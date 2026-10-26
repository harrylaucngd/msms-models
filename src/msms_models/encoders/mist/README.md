# MIST

[`scaled.py`](scaled.py) loads ScaledMIST and computes spectrum-only posteriors; [`__init__.py`](__init__.py) aligns cached posteriors by spectrum ID. The ScaledMIST architecture comes from the pinned `coleygroup/MIST` dependency, not a sibling checkout. [`models/retrieval/mist.py`](../../models/retrieval/mist.py) ranks candidates by these fingerprints; MolForge, MS-GPT and FRIGID are conditioned on them. The MIST released with MS-GPT and FRIGID is not supported (paper §6.1). The `noaug` and `4m-aug` configurations of the `mist` model select different checkpoints.

Public ScaledMIST feature and weight downloads are pending upstream. Use matching local assets or ID-aligned cached posteriors in the meantime.

## Assets

| Configuration key | Input |
|---|---|
| `mist_checkpoint` | Matching ScaledMIST EMA checkpoint; the MSG noaug and 4M-aug configurations select different files. |
| `compact_cache` | Dataset/split-specific spectrum features from the MIST compact archive. |
| `candidates` | ID-keyed candidate pool for retrieval. |

Fingerprint conditioning uses `mist_checkpoint` and `compact_cache`. Alternatively, supply `posterior` and `posterior_ids` to reuse a cache without loading MIST. The [data guide](../../../../docs/data.md#mist-posteriors) defines these formats.

**Parameters.**

| Key | Default | Meaning |
|---|---|---|
| `fingerprint_representation` | `raw` | `binary` thresholds the posterior before ranking; `raw` ranks by the probabilities |
| `fingerprint_threshold` | none | threshold for `binary`; calibrate it for the encoder in use |
| `candidate_ties` | `stable` | `legacy-quicksort` gives the tie order of the original MIST evaluation |
| `encoder_batch_size` | 64 (ScaledMIST), 32 (DiffMS MIST) | spectra per encoder forward pass |
| `attention_padding` | `masked` | DiffMS MIST only; `additive-bool` restores the historical padding mask |

Provider links and current download availability are in the [asset inventory](../../../../docs/assets.md). Paths are explicit in the configurations below.

## Run

From the repository root, after the [common data setup](../../../../docs/data.md) and [asset preparation](../../../../docs/assets.md):

```bash
bash environments/install.sh   # from the repository root
msms-models predict --config configs/mist/msg-noaug.yaml --limit 2 --output results/smoke/mist
msms-models predict --config configs/mist/msg-noaug.yaml
msms-models evaluate --task retrieval --queries results/mist-noaug/msg/queries.jsonl \
  --predictions results/mist-noaug/msg/predictions.jsonl \
  --output results/mist-noaug/msg/metrics.json --mces
```

`--limit 2` exercises the configured inference on the first two queries; it is a plumbing check, not a benchmark result.

## Variants and source

| Model ID | Configurations | Entry point |
|---|---|---|
| `mist` | [msg-noaug](../../../../configs/mist/msg-noaug.yaml), [msg-4m-aug](../../../../configs/mist/msg-4m-aug.yaml) | [models/retrieval/mist.py](../../models/retrieval/mist.py) |

Shared loading, fingerprint ranking and evaluation remain in the library core instead of being copied per method. [Third-party notices](../../../../THIRD_PARTY_NOTICES.md) record the source revisions and licenses.

## DiffMS MIST (vDiffMS diagnostic only)

`diffms.py` and [`vendor/`](vendor/) contain the MIST released with
DiffMS ("vDiffMS"), the encoder of the original MIST + MolForge benchmark. The shared
posterior interface selects it when `diffms_features` is given together with
`mist_checkpoint` (the DiffMS Zenodo `mist_msg.pt`).
`diffms_features` is either a directory of DiffMS subformula trees (`<id>.json`) or a
JSONL bundle of `{"id", "subformulae"}` records. Spectra are encoded in query order,
`encoder_batch_size` consecutive spectra per forward pass, each batch right-padded to
its longest spectrum. With the default `attention_padding: masked` the result does not
depend on the batch size; `additive-bool` reproduces the historical batch-dependent
posteriors at the configured `encoder_batch_size`. Only MassSpecGym is
configured. See the [Table 5 diagnostic rows](../../../../paper/table5/mist_vdiffms_molforge/).
