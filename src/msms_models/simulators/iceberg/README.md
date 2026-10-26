# ICEBERG

[`__init__.py`](__init__.py) loads the generator/intensity pair and predicts sparse spectra; [`models/retrieval/iceberg.py`](../../models/retrieval/iceberg.py) supplies that predictor to the shared candidate ranking in [`simulators/retrieve.py`](../retrieve.py). FRIGID refinement and FOAM reuse the same predictor. `vendor/` contains the generator, intensity predictor, graph layers, fragmentation and chemistry functions required for inference. There is no required external ms-pred checkout for this method. Both matching generator and intensity checkpoints are required.

Queries need a collision energy in eV, adduct, instrument and precursor m/z. The paper rows use the frozen queries with imputed energies. `query_intensity_space` defaults to `native`, which takes the peaks as prepared; `raw` applies one square root to measured intensities.

## Assets

| Configuration key | Input |
|---|---|
| `generator_checkpoint, intensity_checkpoint` | Matching dataset-specific legacy `dag_pred` weights. Both are loaded strictly. |
| `candidates` | The complete ordered retrieval candidate pool. |

**Parameters.**

| Key | Default | Meaning |
|---|---|---|
| `query_intensity_space` | `native` | `raw` applies one square root to the query intensities |
| `num_bins`, `upper_limit` | 150000, 1500 | spectrum binning: number of bins up to `upper_limit` m/z |
| `sparse_k` | 100 | predicted peaks kept per spectrum |
| `max_heavy_atoms` | 10000 | candidates with more heavy atoms get distance 1 |
| `fragment_threshold` | 0.0 | minimum fragment probability in the generator |
| `max_nodes` | 100 | maximum fragments per molecule |

Provider links and current download availability are in the [asset inventory](../../../../docs/assets.md). Paths are explicit in the configurations below.

## Run

From the repository root, after the [common data setup](../../../../docs/data.md) and [asset preparation](../../../../docs/assets.md):

```bash
bash environments/install.sh   # from the repository root
msms-models predict --config paper/table3/iceberg/config.yaml --limit 2 --output results/smoke/iceberg
msms-models predict --config paper/table3/iceberg/config.yaml
msms-models evaluate --protocol table3 --queries results/paper/table3/iceberg/queries.jsonl \
  --predictions results/paper/table3/iceberg/predictions.jsonl \
  --output results/paper/table3/iceberg/metrics.json
```

`--limit 2` exercises the configured inference on the first two queries; it is a plumbing check, not a benchmark result.

## Variants and source

| Model ID | Configurations | Entry point |
|---|---|---|
| `iceberg` | [msg](../../../../paper/table3/iceberg/config.yaml) | [models/retrieval/iceberg.py](../../models/retrieval/iceberg.py) |

Shared loading, fingerprint ranking and evaluation remain in the library core instead of being copied per method. [Third-party notices](../../../../THIRD_PARTY_NOTICES.md) record the source revisions and licenses.
