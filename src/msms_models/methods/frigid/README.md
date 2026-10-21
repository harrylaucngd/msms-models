# FRIGID

`inference.py` implements checkpoint loading and adapters for one-shot generation and refinement. `vendor/` contains the DLM, sampler, masking and formula/length utilities. Refinement shares the [ICEBERG implementation](../iceberg/README.md) and [simulation contract](../../simulation.py).

`frigid-base` is the first scratch round. `num_rounds` counts **all** rounds: ten refinement transitions require `num_rounds: 11`. The CANOPUS configuration uses 24 refinements (`num_rounds: 25`). A length model and its ordered feature list are mandatory; they must match the tokenizer. MSG configurations pin the recovered encoder/decoder hashes and calibrated fingerprint threshold.

## Assets

| Configuration key | Input |
|---|---|
| `decoder_checkpoint` | DLM weights from the official checkpoint archive. |
| `safe_tokenizer` | The SAFE tokenizer paired with that decoder. |
| `length_model, length_features` | Dataset-specific serialized length predictor and its ordered feature list. |
| `generator_checkpoint, intensity_checkpoint` | Matching ICEBERG pair; required by refinement, unused by `frigid-base`. |

Fingerprint conditioning uses `mist_checkpoint` and `compact_cache`. Alternatively, supply `posterior`, `posterior_ids` and `posterior_manifest` to reuse a validated cache without loading MIST. The [data guide](../../../../docs/data.md#mist-compact-features-and-posteriors) defines these formats.

Provider links and current download availability are in the [asset inventory](../../../../docs/assets.md). Paths are explicit in the configurations below.

## Run

From the repository root, after the [common data setup](../../../../docs/data.md) and [asset preparation](../../../../docs/assets.md):

```bash
python -m pip install -e '.[mist,frigid,iceberg]'
msms-models doctor --config configs/frigid/msg-base.yaml
msms-models smoke --config configs/frigid/msg-base.yaml --output results/smoke/frigid-base --limit 2
msms-models infer --config configs/frigid/msg-base.yaml
msms-models evaluate --task de-novo --queries data/msg/queries.jsonl \
  --predictions results/frigid-base/msg/predictions.jsonl \
  --output results/frigid-base/msg/metrics.json --mces
```

The smoke command uses the configured inference budget on a small subset. For a cheaper developmental check, copy the configuration and explicitly reduce the budget; retain that configuration with the report. Choose a fresh output directory on each run.

## Variants and source

| Model ID | Configurations | Entry point |
|---|---|---|
| `frigid-base` | [msg](../../../../configs/frigid/msg-base.yaml), [canopus](../../../../configs/frigid/canopus-base.yaml) | [inference.py](inference.py) |
| `frigid` | [msg](../../../../configs/frigid/msg.yaml), [canopus](../../../../configs/frigid/canopus.yaml) | [inference.py](inference.py) |

[spec.py](spec.py) declares required assets, optional imports and accepted parameters. Shared loading, fingerprint ranking and evaluation remain in the library core instead of being copied per method. [Third-party notices](../../../../THIRD_PARTY_NOTICES.md) record the source revisions and licenses.
