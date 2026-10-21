# FOAM

`inference.py` adapts the native graph optimizer in `vendor/` to the shared query and prediction contracts. Scoring uses the shared [ICEBERG implementation](../../../simulators/iceberg/README.md). The seed list is an explicit input.

Seed generation is an explicit input stage. Supply the chosen generator outputs or a frozen database candidate export, with the target excluded. Different seeds define a different optimization run; live database ordering is not a stable replacement for a frozen export.

## Assets

| Configuration key | Input |
|---|---|
| `generator_checkpoint, intensity_checkpoint` | Matching ICEBERG generator/intensity weights for spectral scoring. |
| `seeds` | ID-keyed initial candidate structures, supplied independently of the target. |

Provider links and current download availability are in the [asset inventory](../../../../../docs/assets.md). Paths are explicit in the configurations below.

## Run

From the repository root, after the [common data setup](../../../../../docs/data.md) and [asset preparation](../../../../../docs/assets.md):

```bash
python -m pip install -e '.[foam,iceberg]'
msms-models predict --config configs/foam/msg.yaml --limit 2 --output results/smoke/foam
msms-models predict --config configs/foam/msg.yaml
msms-models evaluate --task de-novo --queries data/msg/queries.jsonl \
  --predictions results/foam/msg/predictions.jsonl \
  --output results/foam/msg/metrics.json --mces
```

`--limit 2` exercises the configured inference on the first two queries; it is a plumbing check, not a benchmark result. Choose a fresh output directory for each run.

## Variants and source

| Model ID | Configurations | Entry point |
|---|---|---|
| `foam` | [msg](../../../../../configs/foam/msg.yaml), [canopus](../../../../../configs/foam/canopus.yaml) | [inference.py](inference.py) |

Shared loading, fingerprint ranking and evaluation remain in the library core instead of being copied per method. [Third-party notices](../../../../../THIRD_PARTY_NOTICES.md) record the source revisions and licenses.
