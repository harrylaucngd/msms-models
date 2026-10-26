# GLACIER

[`__init__.py`](__init__.py) loads the checkpoint and predicts sparse spectra; [`models/retrieval/glacier.py`](../../models/retrieval/glacier.py) supplies that predictor to the shared candidate ranking in [`simulators/retrieve.py`](../retrieve.py). The matching architecture is `mist.glacier` in the pinned public MIST package; it uses the pinned public ms-pred helpers documented in [installation](../../../../docs/install.md). This dependency boundary is explicit because the released GLACIER checkpoint has layers absent from the historical standard ms-pred architecture.

Queries need a collision energy in eV, adduct, instrument and precursor m/z. The paper rows use the frozen queries with imputed energies. `query_intensity_space` defaults to `native`, which takes the peaks as prepared; `raw` applies one square root to measured intensities.

## Assets

| Configuration key | Input |
|---|---|
| `checkpoint` | GLACIER compatibility checkpoint paired with the pinned MIST implementation. |
| `candidates` | The complete ordered retrieval candidate pool. |

**Parameters.**

| Key | Default | Meaning |
|---|---|---|
| `query_intensity_space` | `native` | `raw` applies one square root to the query intensities |
| `num_bins`, `upper_limit` | 150000, 1500 | spectrum binning: number of bins up to `upper_limit` m/z |
| `sparse_k` | 100 | predicted peaks kept per spectrum |
| `max_heavy_atoms` | 10000 | candidates with more heavy atoms get distance 1 |

Provider links and current download availability are in the [asset inventory](../../../../docs/assets.md). Paths are explicit in the configurations below.

## Run

From the repository root, after the [common data setup](../../../../docs/data.md) and [asset preparation](../../../../docs/assets.md):

```bash
bash environments/install.sh   # from the repository root
msms-models predict --config paper/table3/glacier/config.yaml --limit 2 --output results/smoke/glacier
msms-models predict --config paper/table3/glacier/config.yaml
msms-models evaluate --protocol table3 --queries results/paper/table3/glacier/queries.jsonl \
  --predictions results/paper/table3/glacier/predictions.jsonl \
  --output results/paper/table3/glacier/metrics.json
```

`--limit 2` exercises the configured inference on the first two queries; it is a plumbing check, not a benchmark result.

## Variants and source

| Model ID | Configurations | Entry point |
|---|---|---|
| `glacier` | [msg](../../../../paper/table3/glacier/config.yaml) | [models/retrieval/glacier.py](../../models/retrieval/glacier.py) |

Shared loading, fingerprint ranking and evaluation remain in the library core instead of being copied per method. [Third-party notices](../../../../THIRD_PARTY_NOTICES.md) record the source revisions and licenses.
