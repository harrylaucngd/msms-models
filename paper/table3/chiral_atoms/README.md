# Table 3 · (5) Ranking by number of chiral atoms

| | Hit@1 | Hit@5 | Hit@20 | MCES@1 |
|---|---|---|---|---|
| (5) Ranking by Number of Chiral Atoms | 4.88% (4.36–5.44) | 16.85% (15.92–17.79) | 37.86% (36.72–39.09) | 14.4 (14.23–14.56) |

Intervals are 99.9% BCa over 20,000 spectrum-level resamples.

**Source.** [`models/retrieval/chiral_atoms.py`](../../../src/msms_models/models/retrieval/chiral_atoms.py)

**What runs.** Counts the atoms with an explicit chiral tag (`@`/`@@`) in each candidate SMILES and ranks by that count. The direction is fixed on the training split: descending when true molecules carry more annotated stereocentres than their candidates on average. Ties are broken by a seeded random permutation, because the official pools list the true molecule first. No spectrum is read.

**Inputs.** Official MassSpecGym test queries and original formula candidate pool, plus the training split for the direction (`bash paper/prepare_data.sh`).

**Environment.** Core package (`pip install -e '.[mces]'`). CPU only.

**Bootstrap.** seed 0, manifest order.

**Reproduction status.** Rewritten from the paper's Appendix A.4; the original experiment code was not available. It has not been run at full scale here, so its numbers may differ from the printed ones within the tolerance expected of a reimplementation. `compare.py` reports any difference.

## Run

From the repository root:

```bash
bash paper/table3/chiral_atoms/run.sh
```

`run.sh` predicts, evaluates with the fixed `table3` protocol and checks the printed
values with [`compare.py`](../../compare.py).
