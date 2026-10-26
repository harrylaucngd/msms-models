# Table 3 · (4) PubChem default ranking

| | Hit@1 | Hit@5 | Hit@20 | MCES@1 |
|---|---|---|---|---|
| (4) PubChem Default Ranking | 49.98% (48.72–51.20) | 54.13% (52.93–55.28) | 61.04% (59.83–62.22) | 7.43 (7.22–7.64) |

Intervals are 99.9% BCa over 20,000 spectrum-level resamples.

**Source.** [`models/retrieval/pubchem_order`](../../../src/msms_models/models/retrieval/pubchem_order/)

**What runs.** For each query formula, PubChem PUG-REST (`compound/fastformula`) returns matching compounds in its default order, which favours well-characterized molecules. Candidates are ranked by the first position of their InChIKey-14 in that listing; unlisted candidates follow in random order. No spectrum is read.

**Inputs.** Official test queries and formula candidate pool, plus the PubChem listing `data/paper/table3/pubchem_order.jsonl`. It is not distributed; `run.sh` creates it with `python -m msms_models.models.retrieval.pubchem_order.build` when it is missing. PubChem's order changes over time, so a new listing can give a different result from the paper's.

**Environment.** The locked environment (`bash environments/install.sh`; see [installation](../../../docs/install.md)). CPU only. Building the listing needs internet access (about one request per formula, at most five per second).

## Run

From the repository root:

```bash
bash paper/table3/pubchem_order/run.sh
```

Set `OUT=...` to choose the output directory.
`run.sh` predicts and evaluates with the fixed `table3` protocol.
