# Nearest Neighbor

Ranks candidates by the fingerprints of the training spectra most similar to the query.
Spectra are annotated with subformulae and turned into sparse peak tokens;
[`index.py`](index.py) builds an index of the training spectra, chooses the neighbour
settings on validation, and saves and loads it. [`inference.py`](inference.py) scores each
query against the index, builds a consensus fingerprint from its neighbours and ranks the
full candidate pool; training spectra are not filtered by formula. `vendor/features.py` contains the peak annotation, sparse tokens and
consensus functions.

## Assets

| Configuration key | Input |
|---|---|
| `nn_index` | The directory written by `fit-nn`: feature vocabulary, training fingerprints and selected settings. |
| `candidates` | Ordered candidate pools prepared with `prepare-candidates`. |

**Parameters.** `nn_batch_size` (16) is the number of queries scored per batch.
`candidate_ties` (`stable`) keeps the supplied candidate order on ties. The neighbour
settings (w, α, T, k) are stored in the index by `fit-nn`; the selection for the paper row is
`w=2, alpha=2, T=0.1, k=50`.

## Run

From the repository root, after `bash paper/prepare_data.sh`:

```bash
bash paper/table3/nearest_neighbor/run.sh
```

`run.sh` fits the index first if it is missing. Fitting searches the published grid on
validation and can take substantial CPU time:

```bash
msms-models fit-nn --train data/msg/train.jsonl --validation data/msg/val.jsonl \
  --output data/indices/nearest-neighbor-msg
```

## Source

[serenaklm/ML_MS_analysis](https://github.com/serenaklm/ML_MS_analysis) at `6817e9bd994f`,
MIT; see [`vendor/LICENSE.txt`](vendor/LICENSE.txt) and the
[third-party notices](../../../../../THIRD_PARTY_NOTICES.md).
