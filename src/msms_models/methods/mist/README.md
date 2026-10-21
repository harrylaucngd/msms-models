# MIST

`inference.py` ranks candidate fingerprints. [encoder.py](encoder.py) loads the encoder and computes spectrum-only posteriors; [posteriors.py](../../posteriors.py) validates and aligns cached fingerprints. Encoder architecture comes from the pinned `coleygroup/MIST` dependency, not a sibling checkout. The `mist-noaug` and `mist-4m-aug` IDs select different assets.

Public ScaledMIST feature and weight downloads are pending upstream. Use matching local assets or ID-aligned cached posteriors in the meantime.

## Assets

| Configuration key | Input |
|---|---|
| `mist_checkpoint` | Matching ScaledMIST EMA checkpoint; the MSG noaug and 4M-aug configurations select different files. |
| `compact_cache` | Dataset/split-specific spectrum features from the MIST compact archive. |
| `candidates` | ID-keyed candidate pool for retrieval. |

Fingerprint conditioning uses `mist_checkpoint` and `compact_cache`. Alternatively, supply `posterior`, `posterior_ids` and `posterior_manifest` to reuse a validated cache without loading MIST. The [data guide](../../../../docs/data.md#mist-compact-features-and-posteriors) defines these formats.

Provider links and current download availability are in the [asset inventory](../../../../docs/assets.md). Paths are explicit in the configurations below.

## Run

From the repository root, after the [common data setup](../../../../docs/data.md) and [asset preparation](../../../../docs/assets.md):

```bash
python -m pip install -e '.[mist]'
msms-models doctor --config configs/mist/msg-noaug.yaml
msms-models smoke --config configs/mist/msg-noaug.yaml --output results/smoke/mist-noaug --limit 2
msms-models infer --config configs/mist/msg-noaug.yaml
msms-models evaluate --task retrieval --queries data/msg/queries.jsonl \
  --predictions results/mist-noaug/msg/predictions.jsonl \
  --output results/mist-noaug/msg/metrics.json --mces
```

The smoke command uses the configured inference budget on a small subset. For a cheaper developmental check, copy the configuration and explicitly reduce the budget; retain that configuration with the report. Choose a fresh output directory on each run.

## Variants and source

| Model ID | Configurations | Entry point |
|---|---|---|
| `mist-noaug` | [msg](../../../../configs/mist/msg-noaug.yaml) | [inference.py](inference.py) |
| `mist-4m-aug` | [msg](../../../../configs/mist/msg-4m-aug.yaml) | [inference.py](inference.py) |

[spec.py](spec.py) declares required assets, optional imports and accepted parameters. Shared loading, fingerprint ranking and evaluation remain in the library core instead of being copied per method. [Third-party notices](../../../../THIRD_PARTY_NOTICES.md) record the source revisions and licenses.

## Historical benchmark encoders

`legacy.py` and `vendor/` contain the MIST architecture and annotated-spectrum
featurization released with MS-GPT. The shared posterior interface selects this
path when `legacy_features` points to the released parquet inputs. This path
reads no target structures or fingerprints and checks spectrum IDs and formulas.
It preserves the upstream MS-GPT posterior computation, including its bypassed
intermediate projection and zero instrument category; it is a distinct model
from the ScaledMIST checkpoints above. Unlike upstream, padded peaks are excluded from attention
(paper Sec. 6.1.1); set `attention_padding: additive-bool` only to reproduce the
historical batch-dependent posteriors. The pinned ScaledMIST package already
applies this fix. See the [MS-GPT guide](../ms_gpt/README.md)
for downloads, hashes and matching calibrations.
