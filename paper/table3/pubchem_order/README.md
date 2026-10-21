# Table 3 · (4) PubChem default ranking

| | Hit@1 | Hit@5 | Hit@20 | MCES@1 |
|---|---|---|---|---|
| (4) PubChem Default Ranking | 49.98% (48.72–51.20) | 54.13% (52.93–55.28) | 61.04% (59.83–62.22) | 7.43 (7.22–7.64) |

Intervals are 99.9% BCa over 20,000 spectrum-level resamples.

**Source.** [`models/retrieval/pubchem_order`](../../../src/msms_models/models/retrieval/pubchem_order/)

**What runs.** For each query formula, PubChem PUG-REST (`compound/fastformula`) returns matching compounds in its default order, which favours well-characterized molecules. Candidates are ranked by the first position of their InChIKey-14 in that listing; unlisted candidates follow in random order. No spectrum is read.

**Inputs.** Official test queries and formula candidate pool, plus a frozen listing `data/paper/table3/pubchem_order.jsonl`. `build.py` creates it, but PubChem's order changes over time, so the released frozen file defines a reported result.

**Environment.** Core package. Building the listing needs internet access (about one request per formula, at most five per second).

**Bootstrap.** seed 0, manifest order.

**Reproduction status.** Rewritten from the paper's Appendix A.4; the original experiment code was not available. It has not been run at full scale here, so its numbers may differ from the printed ones within the tolerance expected of a reimplementation. `compare.py` reports any difference.

## Run

From the repository root:

```bash
bash paper/table3/pubchem_order/run.sh
```

`run.sh` predicts, evaluates with the fixed `table3` protocol and checks the printed
values with [`compare.py`](../../compare.py).
