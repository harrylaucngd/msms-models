# ICEBERG

[`__init__.py`](__init__.py) loads the generator/intensity pair and predicts sparse spectra; [`models/retrieval/iceberg.py`](../../models/retrieval/iceberg.py) supplies that predictor to the shared candidate ranking in [`simulators/retrieve.py`](../retrieve.py). FRIGID refinement and FOAM reuse the same predictor. `vendor/` contains the generator, intensity predictor, graph layers, fragmentation and chemistry functions required for inference. There is no required external ms-pred checkout for this method. Both matching generator and intensity checkpoints are required.

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
msms-models doctor --config paper/table3/iceberg/config.yaml
msms-models predict --config paper/table3/iceberg/config.yaml --limit 2 --output results/smoke/iceberg
msms-models predict --config paper/table3/iceberg/config.yaml
msms-models evaluate --task retrieval --queries data/msg/queries.jsonl \
  --predictions results/iceberg/msg/predictions.jsonl \
  --output results/iceberg/msg/metrics.json --mces
```

`--limit 2` exercises the configured inference on the first two queries; it is a plumbing check, not a benchmark result. Choose a fresh output directory for each run.

## Variants and source

| Model ID | Configurations | Entry point |
|---|---|---|
| `iceberg` | [msg](../../../../paper/table3/iceberg/config.yaml) | [models/retrieval/iceberg.py](../../models/retrieval/iceberg.py) |

Assets, parameters and runtime imports are registered in [models/\_\_init\_\_.py](../../models/__init__.py). Shared loading, fingerprint ranking and evaluation remain in the library core instead of being copied per method. [Third-party notices](../../../../THIRD_PARTY_NOTICES.md) record the source revisions and licenses.
