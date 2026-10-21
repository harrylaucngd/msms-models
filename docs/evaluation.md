# Evaluation protocol

Always evaluate against the original query manifest. For the standard
MassSpecGym split this contains **17,556 test spectra**. Prediction files cannot
change the denominator: missing rows become empty predictions, and duplicate or
unexpected IDs are errors. A supplied subset is evaluated as that subset and
must not be described as the full benchmark.

| Quantity | Definition |
|---|---|
| Hit@k / accuracy@k | Any candidate among the first k has the target's InChIKey connectivity block |
| Structural Tanimoto | Radius-2, 2,048-bit Morgan fingerprint, chirality disabled |
| Retrieval Tanimoto / MCES | Top-ranked candidate only |
| De novo Tanimoto@k / MCES@k | Maximum Tanimoto / minimum MCES among the first k |
| Confidence interval | BCa bootstrap of per-spectrum values; 20,000 resamples, 99.9%, seed 0 |

Fingerprints are computed directly on the input molecular graph. An extra
SMILES round-trip can change explicit hydrogen representation and therefore
Morgan bits, even with chirality disabled. MCES separately removes
stereochemistry and canonicalizes SMILES, preserving isotopes. The default does
not canonicalize tautomers or strip salts. `--canonical-tautomers` declares a
separate normalization protocol; do not combine its numbers with default results.

`preserve` is the default candidate policy. Invalid de novo structures and
disconnected molecules retain their rank slots but earn no match or Tanimoto
credit. An absent/invalid candidate receives MCES penalty `2 × target bond count`.
`--candidate-policy unique-valid` explicitly deduplicates valid connectivity
identities before de novo scoring; it is not permitted for retrieval.

Native decoders already perform method-specific postprocessing. For example,
MS-GPT filters molecular formulas and ranks connectivity groups by frequency,
and FRIGID uses native formula-priority and fingerprint ranking. The evaluator
preserves each model's output order. Retrieval ranks the original supplied pool with stable ties. Use the
same ordering for Hit, top-1 Tanimoto and MCES.

MCES uses `myopic-mces==1.0.1`, threshold 15, the stronger bound, CBC and a
five-second time limit per ordered molecule pair. This is the benchmark's
bounded computation, not a promise of an unrestricted exact graph search.
Solver errors propagate rather than becoming favorable scores. The optional
SQLite cache keys results by molecule pair and solver protocol; successfully
completed pairs survive interruption. `--mces-threads` defaults to one.

Constant metric arrays receive a degenerate interval at their constant value.
Other undefined BCa intervals are errors. `--bootstrap-resamples 0` skips
intervals for diagnostics; report that choice if sharing such results.

The CLI exposes `--mces-solver` (default `PULP_CBC_CMD`). For a native CBC executable
on PATH, select `COIN_CMD`; this is needed when the bundled PuLP binary does not
match the CPU architecture. See [installation](install.md#native-cbc-on-macos).
