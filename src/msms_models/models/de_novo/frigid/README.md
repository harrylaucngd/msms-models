# FRIGID

`inference.py` implements checkpoint loading and adapters for one-shot generation and refinement. `vendor/` contains the DLM, sampler, masking and formula/length utilities. Refinement shares the [ICEBERG simulator](../../../simulators/iceberg/README.md) and the [metadata contract](../../../simulators/retrieve.py).

`frigid-base` is the first scratch round. `num_rounds` counts **all** rounds: ten refinement transitions require `num_rounds: 11`. The CANOPUS configuration uses 24 refinements (`num_rounds: 25`). A length model and its ordered feature list are mandatory; they must match the tokenizer. MSG configurations pin the recovered encoder/decoder hashes and calibrated fingerprint threshold.

## Assets

| Configuration key | Input |
|---|---|
| `decoder_checkpoint` | DLM weights from the official checkpoint archive. |
| `safe_tokenizer` | The SAFE tokenizer paired with that decoder. |
| `length_model, length_features` | Dataset-specific serialized length predictor and its ordered feature list. |
| `generator_checkpoint, intensity_checkpoint` | Matching ICEBERG pair; required by refinement, unused by `frigid-base`. |

Fingerprint conditioning uses `mist_checkpoint` and `compact_cache`. Alternatively, supply `posterior` and `posterior_ids` to reuse a cache without loading MIST. The [data guide](../../../../../docs/data.md#mist-compact-features-and-posteriors) defines these formats.

Provider links and current download availability are in the [asset inventory](../../../../../docs/assets.md). Paths are explicit in the configurations below.

## Run

From the repository root, after the [common data setup](../../../../../docs/data.md) and [asset preparation](../../../../../docs/assets.md):

```bash
python -m pip install -e '.[mist,frigid,iceberg]'
msms-models predict --config paper/table5/frigid_base/config.yaml --limit 2 --output results/smoke/frigid-base
msms-models predict --config paper/table5/frigid_base/config.yaml
msms-models evaluate --task de-novo --queries data/msg/queries.jsonl \
  --predictions results/frigid-base/msg/predictions.jsonl \
  --output results/frigid-base/msg/metrics.json --mces
```

`--limit 2` exercises the configured inference on the first two queries; it is a plumbing check, not a benchmark result. Choose a fresh output directory for each run.

## Variants and source

| Model ID | Configurations | Entry point |
|---|---|---|
| `frigid-base` | [msg](../../../../../paper/table5/frigid_base/config.yaml), [canopus](../../../../../configs/frigid/canopus-base.yaml) | [inference.py](inference.py) |
| `frigid` | [msg](../../../../../paper/table5/frigid/config.yaml), [canopus](../../../../../configs/frigid/canopus.yaml) | [inference.py](inference.py) |

Shared loading, fingerprint ranking and evaluation remain in the library core instead of being copied per method. [Third-party notices](../../../../../THIRD_PARTY_NOTICES.md) record the source revisions and licenses.
