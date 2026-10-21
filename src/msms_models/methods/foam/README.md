# FOAM

`inference.py` adapts the native graph optimizer in `vendor/` to the shared query and prediction contracts. Scoring uses the shared [ICEBERG implementation](../iceberg/README.md). The seed list and its provenance manifest are explicit inputs.

Seed generation is an explicit input stage. Supply the chosen generator outputs and frozen database candidate export, with the target-exclusion policy and sources recorded in the seed manifest. Different seeds define a different optimization run; live database ordering is not a stable replacement for a frozen export.

## Assets

| Configuration key | Input |
|---|---|
| `generator_checkpoint, intensity_checkpoint` | Matching ICEBERG generator/intensity weights for spectral scoring. |
| `seeds` | ID-keyed initial candidate structures, supplied independently of the target. |
| `seed_manifest` | Dataset, seed sources and target-exclusion provenance. |

Provider links and current download availability are in the [asset inventory](../../../../docs/assets.md). Paths are explicit in the configurations below.

## Run

From the repository root, after the [common data setup](../../../../docs/data.md) and [asset preparation](../../../../docs/assets.md):

```bash
python -m pip install -e '.[foam,iceberg]'
msms-models doctor --config configs/foam/msg.yaml
msms-models smoke --config configs/foam/msg.yaml --output results/smoke/foam --limit 2
msms-models infer --config configs/foam/msg.yaml
msms-models evaluate --task de-novo --queries data/msg/queries.jsonl \
  --predictions results/foam/msg/predictions.jsonl \
  --output results/foam/msg/metrics.json --mces
```

The smoke command uses the configured inference budget on a small subset. For a cheaper developmental check, copy the configuration and explicitly reduce the budget; retain that configuration with the report. Choose a fresh output directory on each run.

## Variants and source

| Model ID | Configurations | Entry point |
|---|---|---|
| `foam` | [msg](../../../../configs/foam/msg.yaml), [canopus](../../../../configs/foam/canopus.yaml) | [inference.py](inference.py) |

[spec.py](spec.py) declares required assets, optional imports and accepted parameters. Shared loading, fingerprint ranking and evaluation remain in the library core instead of being copied per method. [Third-party notices](../../../../THIRD_PARTY_NOTICES.md) record the source revisions and licenses.
