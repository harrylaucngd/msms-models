# GLACIER

[`__init__.py`](__init__.py) loads the checkpoint and predicts sparse spectra; [`models/retrieval/glacier.py`](../../models/retrieval/glacier.py) supplies that predictor to the shared candidate ranking in [`simulators/retrieve.py`](../retrieve.py). The matching architecture is `mist.glacier` in the pinned public MIST package; it uses the pinned public ms-pred helpers documented in [installation](../../../../docs/install.md). This dependency boundary is explicit because the released GLACIER checkpoint has layers absent from the historical standard ms-pred architecture.

Inputs must include collision energy in eV, adduct, instrument and precursor m/z. Set `query_intensity_space` explicitly. `raw` applies one square root to measured intensities; `native` accepts already prepared peaks. Missing energies require an explicitly supplied metadata preparation step.

## Assets

| Configuration key | Input |
|---|---|
| `checkpoint` | GLACIER compatibility checkpoint paired with the pinned MIST implementation. |
| `candidates` | The complete ordered retrieval candidate pool. |

Provider links and current download availability are in the [asset inventory](../../../../docs/assets.md). Paths are explicit in the configurations below.

## Run

From the repository root, after the [common data setup](../../../../docs/data.md) and [asset preparation](../../../../docs/assets.md):

```bash
python -m pip install -e '.[mist,iceberg,glacier]'
msms-models predict --config paper/table3/glacier/config.yaml --limit 2 --output results/smoke/glacier
msms-models predict --config paper/table3/glacier/config.yaml
msms-models evaluate --task retrieval --queries data/msg/queries.jsonl \
  --predictions results/glacier/msg/predictions.jsonl \
  --output results/glacier/msg/metrics.json --mces
```

`--limit 2` exercises the configured inference on the first two queries; it is a plumbing check, not a benchmark result. Choose a fresh output directory for each run.

## Variants and source

| Model ID | Configurations | Entry point |
|---|---|---|
| `glacier` | [msg](../../../../paper/table3/glacier/config.yaml) | [models/retrieval/glacier.py](../../models/retrieval/glacier.py) |

Shared loading, fingerprint ranking and evaluation remain in the library core instead of being copied per method. [Third-party notices](../../../../THIRD_PARTY_NOTICES.md) record the source revisions and licenses.
