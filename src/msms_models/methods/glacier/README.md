# GLACIER

`inference.py` owns checkpoint loading and sparse spectrum prediction, and supplies its predictor to the shared candidate ranking in [simulation.py](../../simulation.py). The matching architecture is `mist.glacier` in the pinned public MIST package; it uses the pinned public ms-pred helpers documented in [installation](../../../../docs/install.md). This dependency boundary is explicit because the released GLACIER checkpoint has layers absent from the historical standard ms-pred architecture.

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
msms-models doctor --config configs/glacier/msg.yaml
msms-models smoke --config configs/glacier/msg.yaml --output results/smoke/glacier --limit 2
msms-models infer --config configs/glacier/msg.yaml
msms-models evaluate --task retrieval --queries data/msg/queries.jsonl \
  --predictions results/glacier/msg/predictions.jsonl \
  --output results/glacier/msg/metrics.json --mces
```

The smoke command uses the configured inference budget on a small subset. For a cheaper developmental check, copy the configuration and explicitly reduce the budget; retain that configuration with the report. Choose a fresh output directory on each run.

## Variants and source

| Model ID | Configurations | Entry point |
|---|---|---|
| `glacier` | [msg](../../../../configs/glacier/msg.yaml) | [inference.py](inference.py) |

[spec.py](spec.py) declares required assets, optional imports and accepted parameters. Shared loading, fingerprint ranking and evaluation remain in the library core instead of being copied per method. [Third-party notices](../../../../THIRD_PARTY_NOTICES.md) record the source revisions and licenses.
