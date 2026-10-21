# Evaluation

```bash
msms-models evaluate --protocol table3 --queries data/msg/queries.jsonl \
  --predictions results/run/predictions.jsonl --output results/run/metrics.json
```

Evaluation always uses the original query file, which holds 17,556 test spectra
for MassSpecGym:
- A missing prediction counts as an empty one.
- The output holds the means, intervals and protocol. A `.details.jsonl` file next to it holds the per-spectrum values.

## Metrics

These follow the MassSpecGym definitions (paper Appendix A.6):

| Metric | Definition |
|---|---|
| Hit@k / Acc@k | some candidate in the top k has the target's InChIKey-14 |
| Tanimoto | Morgan radius 2, 2,048 bits, no chirality, on the input graph |
| MCES | myopic MCES 1.0.1, threshold 15, `always_stronger_bound=True`, CBC, 5 s per pair |
| Retrieval Tanimoto and MCES | top-1 candidate only |
| De novo Tanimoto@k / MCES@k | best value among the first k slots |

- **Invalid or missing slots.** They keep their position and score no hit and Tanimoto 0. Their MCES is **100**, and each spectrum's minimum is capped at 100, as in MassSpecGym.
- **Disconnected SMILES.** These are valid molecules.
- **Stereochemistry.** MCES removes it before comparing; isotopes are kept. Tautomers and salts are not normalized unless `--canonical-tautomers` is given.

## Protocols

`--protocol` fixes the metrics of a paper table; any explicit option overrides a single
field:

| Protocol | Task | Cutoffs | MCES |
|---|---|---|---|
| `table3` | retrieval | 1, 5, 20 | @1 |
| `table5` | de novo | 1, 10 | @1, @10 |

**Intervals.** These are 99.9% BCa bootstraps over 20,000 spectrum-level resamples.
`--bootstrap-resamples 0` skips them.

**Candidate policy.** `preserve` (the default) scores the model's own order, and that
order is required for retrieval. `--candidate-policy unique-valid` drops invalid and
repeated de novo structures before scoring. That is a different protocol, so do not
mix its numbers with `preserve` results.

## MCES in practice

- **Cache.** `--mces-cache file.sqlite` stores finished pairs, so an interrupted evaluation resumes where it stopped.
- **macOS.** PuLP's bundled CBC can target the wrong CPU there. Install a native CBC and pass `--mces-solver COIN_CMD`.
