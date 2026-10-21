# ICEBERG

`inference.py` owns checkpoint loading and sparse spectrum prediction, and supplies its predictor to the shared candidate ranking in [simulation.py](../../simulation.py). `vendor/` contains the generator, intensity predictor, graph layers, fragmentation and chemistry functions required for inference. There is no required external ms-pred checkout for this method. Both matching generator and intensity checkpoints are required.

Inputs must include collision energy in eV, adduct, instrument and precursor m/z. Set `query_intensity_space` explicitly. `raw` applies one square root to measured intensities; `native` accepts already prepared peaks. Missing energies require an explicitly supplied metadata preparation step.

## Assets

| Configuration key | Input |
|---|---|
| `generator_checkpoint, intensity_checkpoint` | Matching dataset-specific legacy `dag_pred` weights. Both are loaded strictly. |
| `candidates` | The complete ordered retrieval candidate pool. |

Provider links and current download availability are in the [asset inventory](../../../../docs/assets.md). Paths are explicit in the configurations below.

## Run

From the repository root, after the [common data setup](../../../../docs/data.md) and [asset preparation](../../../../docs/assets.md):

```bash
python -m pip install -e '.[iceberg]'
msms-models doctor --config configs/iceberg/msg.yaml
msms-models smoke --config configs/iceberg/msg.yaml --output results/smoke/iceberg --limit 2
msms-models infer --config configs/iceberg/msg.yaml
msms-models evaluate --task retrieval --queries data/msg/queries.jsonl \
  --predictions results/iceberg/msg/predictions.jsonl \
  --output results/iceberg/msg/metrics.json --mces
```

The smoke command uses the configured inference budget on a small subset. For a cheaper developmental check, copy the configuration and explicitly reduce the budget; retain that configuration with the report. Choose a fresh output directory on each run.

## Variants and source

| Model ID | Configurations | Entry point |
|---|---|---|
| `iceberg` | [msg](../../../../configs/iceberg/msg.yaml) | [inference.py](inference.py) |

[spec.py](spec.py) declares required assets, optional imports and accepted parameters. Shared loading, fingerprint ranking and evaluation remain in the library core instead of being copied per method. [Third-party notices](../../../../THIRD_PARTY_NOTICES.md) record the source revisions and licenses.
