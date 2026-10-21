# Nearest Neighbor

`features.py` contains peak annotation, sparse tokens and native consensus functions. `index.py` builds a train-only index, chooses parameters on validation, and saves/loads it. `inference.py` ranks the full candidate pool. No training-spectrum formula filter is applied.

Prepare the official training and validation splits in addition to test, then fit once:

```bash
msms-models prepare --input data/MassSpecGym.tsv --split train --output data/msg/train.jsonl
msms-models prepare --input data/MassSpecGym.tsv --split val --output data/msg/val.jsonl
msms-models fit-nn --train data/msg/train.jsonl --validation data/msg/val.jsonl --output data/indices/subformula-msg
```

Full validation searches the published grid and can take substantial CPU time. The paper selected `w=2, alpha=2, T=0.1, k=50`; the implementation recomputes selection rather than embedding test-selected values. Use the pinned `[nn]` environment for neighbor tie parity.

## Assets

| Configuration key | Input |
|---|---|
| `nn_index` | Build from training spectra and validation selection with `fit-nn`. The directory includes its feature vocabulary, fingerprints and selected parameters. |
| `candidates` | Ordered candidate pools prepared with `prepare-candidates`. |

Provider links and current download availability are in the [asset inventory](../../../../docs/assets.md). Paths are explicit in the configurations below.

## Run

From the repository root, after the [common data setup](../../../../docs/data.md) and [asset preparation](../../../../docs/assets.md):

```bash
python -m pip install -e '.[nn]'
msms-models doctor --config configs/nearest_neighbor/msg.yaml
msms-models smoke --config configs/nearest_neighbor/msg.yaml --output results/smoke/subformula-nn --limit 2
msms-models infer --config configs/nearest_neighbor/msg.yaml
msms-models evaluate --task retrieval --queries data/msg/queries.jsonl \
  --predictions results/subformula-nn/msg/predictions.jsonl \
  --output results/subformula-nn/msg/metrics.json --mces
```

The smoke command uses the configured inference budget on a small subset. For a cheaper developmental check, copy the configuration and explicitly reduce the budget; retain that configuration with the report. Choose a fresh output directory on each run.

## Variants and source

| Model ID | Configurations | Entry point |
|---|---|---|
| `subformula-nn` | [msg](../../../../configs/nearest_neighbor/msg.yaml) | [inference.py](inference.py) |

[spec.py](spec.py) declares required assets, optional imports and accepted parameters. Shared loading, fingerprint ranking and evaluation remain in the library core instead of being copied per method. [Third-party notices](../../../../THIRD_PARTY_NOTICES.md) record the source revisions and licenses.
