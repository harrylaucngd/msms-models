# Nearest Neighbor

`vendor/features.py` contains peak annotation, sparse tokens and native consensus functions. `index.py` builds a train-only index, chooses parameters on validation, and saves/loads it. `inference.py` ranks the full candidate pool. No training-spectrum formula filter is applied.

Prepare the official training and validation splits in addition to test, then fit once:

```bash
msms-models prepare --input data/MassSpecGym.tsv --split train --output data/msg/train.jsonl
msms-models prepare --input data/MassSpecGym.tsv --split val --output data/msg/val.jsonl
msms-models fit-nn --train data/msg/train.jsonl --validation data/msg/val.jsonl --output data/indices/nearest-neighbor-msg
```

Full validation searches the published grid and can take substantial CPU time. The paper selected `w=2, alpha=2, T=0.1, k=50`; the implementation recomputes selection rather than embedding test-selected values.

## Assets

| Configuration key | Input |
|---|---|
| `nn_index` | Build from training spectra and validation selection with `fit-nn`. The directory includes its feature vocabulary, fingerprints and selected parameters. |
| `candidates` | Ordered candidate pools prepared with `prepare-candidates`. |

**Parameters.** `nn_batch_size` (16) is the number of queries scored per batch. `candidate_ties` (`stable`) keeps the supplied candidate order on ties. The neighbour settings (w, α, T, k) are stored in the index by `fit-nn`.

Provider links and current download availability are in the [asset inventory](../../../../../docs/assets.md). Paths are explicit in the configurations below.

## Run

From the repository root, after the [common data setup](../../../../../docs/data.md) and [asset preparation](../../../../../docs/assets.md):

```bash
bash environments/install.sh   # from the repository root
msms-models predict --config paper/table3/nearest_neighbor/config.yaml --limit 2 --output results/smoke/nearest-neighbor
msms-models predict --config paper/table3/nearest_neighbor/config.yaml
msms-models evaluate --protocol table3 --queries results/paper/table3/nearest_neighbor/queries.jsonl \
  --predictions results/paper/table3/nearest_neighbor/predictions.jsonl \
  --output results/paper/table3/nearest_neighbor/metrics.json
```

`--limit 2` exercises the configured inference on the first two queries; it is a plumbing check, not a benchmark result.

## Variants and source

| Model ID | Configurations | Entry point |
|---|---|---|
| `nearest-neighbor` | [msg](../../../../../paper/table3/nearest_neighbor/config.yaml) | [inference.py](inference.py) |

Shared loading, fingerprint ranking and evaluation remain in the library core instead of being copied per method. [Third-party notices](../../../../../THIRD_PARTY_NOTICES.md) record the source revisions and licenses.
