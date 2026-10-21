# Validation

Validate three separate properties: implementation agreement with the official
method, compatibility of the selected weights and inputs, and benchmark accuracy.
Passing an inference smoke test does not establish full-test accuracy.

## Checks performed on 2026-10-20

All eight implemented method families were exercised through the normal library
interfaces with real spectra and model assets. Neural runs used isolated
environments on `molgpu03`; the constraints and platform-specific installation
steps are in [installation](install.md). These checks did not regenerate Tables
3 and 5 in full.

| Method | Actual check | Scope |
|---|---|---|
| Nearest Neighbor | Fit training index, select settings on validation, save/load, retrieve and evaluate | Small real MSG subset; reduced explicit parameter grid |
| MIST | GPU encoding and fingerprint retrieval with the portable augmented checkpoint | Two MSG queries; thresholded bits exactly match archived MolForge inputs |
| MolForge | Full MIST → decoder → evaluation on GPU | Two MSG and two CANOPUS-family queries; ten raw beams per query |
| MS-GPT | Historical MIST → released decoder → exact LoRA merge → generation/evaluation on GPU | MSG adapted model and NPLIB1 base model; default 20 × 5 generation also checked |
| FRIGID-base | Released DLM, SAFE tokenizer, MIST encoding and generation on GPU | Two queries, eight proposals; newly fitted auxiliary length model |
| FRIGID | DLM generation, ICEBERG scoring, error-guided refinement and evaluation | Two queries, two rounds, four proposals, K=1/M=2; two real forward predictions |
| FOAM | Released ICEBERG pair and formula-constrained graph search | Two queries, first four frozen official formula candidates as seeds, four-call optimization budget |
| ICEBERG | Strict checkpoint loading, spectrum prediction, retrieval and evaluation | Two measured-energy MSG spectra and a small explicit candidate pool, CPU |
| GLACIER | Strict portable checkpoint loading, spectrum prediction, retrieval and evaluation | The same two measured-energy spectra, CPU |

FRIGID's public archive does not contain the original NGBoost length models.
Its runtime check used a separate length predictor fitted on 512 official MSG
training structures with seed 42. This is an explicit validation substitution,
not the paper predictor. Refinement/FOAM used square-rooted smoke input peaks;
the frozen paper preprocessing and full budgets were not recreated.

Numerical agreement checks:

- The historical MIST posterior path matched the official MS-GPT implementation
  at revision `f2c844e61f079051baf3014c33df5ba2f0524363`: maximum absolute
  difference **0** across two 4096-dimensional outputs.
- Full MIST → MolForge inference reproduced all twenty archived raw beam strings
  for the two MSG queries. Maximum beam-cost differences were below **1.7e-5**.
- The common evaluator reproduced the printed Acc@1/10 and Tanimoto@1/10 from
  **17,556 saved MolForge predictions**. This aggregation check did not regenerate
  every prediction or recompute MCES/confidence intervals.
- A real native CBC solver returned MCES 0 for an identical pair and 1 for a
  one-bond difference on macOS; the evaluator exposes `--mces-solver COIN_CMD`.

Core tests on macOS: **55 passed, 6 optional tests skipped**. The neural environment
on Linux: **59 passed, 2 optional tests skipped**, including nonzero LoRA merge
agreement, the published `CUSTOM` metadata variant, cached/full-prefix generation
agreement, SAFE round-trip and grouped-parquet ID alignment. Model discovery and
cached-fingerprint retrieval work without neural imports. The source distribution
and wheel were built and the wheel was inspected in an isolated installation.

The inspected public MS-GPT exports prepare to 17,556 MSG queries and 672 NPLIB1
queries. The latter is distinct from the archived 819-spectrum CANOPUS evaluation.
Generation can return no formula-valid candidates for a query; such rows remain
in the evaluation denominator.

## Official implementation references

The [method references](references.md) link each paper and its official repository.
[Third-party notices](../THIRD_PARTY_NOTICES.md) identify source revisions, licenses
and the adaptations made for this package. Compare the following outputs with
those implementations using the same weights, inputs, dependencies and random seed:

| Component | Comparison |
|---|---|
| Sub-formula NN — ML_MS_analysis | Peak annotations, sparse features, selected neighbors and consensus fingerprints; preserve index outputs after save/load |
| MIST — coleygroup/MIST | Spectrum-only posteriors, selected checkpoint and spectrum ID alignment |
| MolForge — knu-lcbc/MolForge | Tokenized fingerprints, beam candidates and beam scores |
| MS-GPT-base — VIKI623/MS-GPT | SAFE decoder weights, density-band queries, generated tokens and candidate ordering |
| MS-GPT — VIKI623/MS-GPT | Base checks plus exact adapter tensors, merged model parameters and generated tokens against PEFT inference |
| FRIGID-base / FRIGID — coleygroup/FRIGID | First-round generation, length prediction, formula/fingerprint ordering and spectral refinement rounds |
| ICEBERG / GLACIER — coleygroup/ms-pred | Checkpoint architecture, instrument vocabulary, simulated spectra and entropy-distance ordering |
| FOAM — coleygroup/foam | Mutation/crossover, objective values and final spectral-similarity ordering |

Use the upstream tokenizer and benchmark-specific assets. MS-GPT adapters must
match their MIST encoder and density calibration. Strict weight loading and adapter
tensor checks reject incomplete or incompatible files. The `doctor` command checks
files and dependency availability; it does not load model weights.

## Automated tests

Install the core test dependencies and run:

```bash
python -m pip install -e '.[test,mces]'
ruff check src tests
python -m pytest
python -m build
```

Core CI covers Python 3.10 and 3.11. Tests exercise configuration paths, portable
NN indices, split isolation, prediction alignment, candidate ranking, explicit
hydrogens in Morgan fingerprints, metric denominators, MCES invocation and cache
reuse. MS-GPT adapter tests additionally check nonzero LoRA effects and reject
missing learned tensors when the optional neural dependencies are installed:

```bash
python -m pip install -e '.[test,ms-gpt]'
python -m pytest tests/test_ms_gpt_adapter.py tests/test_decoder_protocols.py
```

These adapter tests use small synthetic models and require no downloaded weights.
Graph/CUDA backends and full-size neural checkpoints require separate inference
checks in their documented environments; the core CI does not exercise them.

## Benchmark validation

For MassSpecGym, evaluate against the official split and candidate pools, keeping
all 17,556 test spectra in the denominator. Use the [evaluation protocol](evaluation.md)
for Hit@1/5/20, structure Tanimoto, bounded MCES and confidence intervals. Check
per-spectrum values before comparing aggregate results. An MCES aggregation check
using cached values is distinct from recomputing the underlying molecule pairs.

Record the checkpoint, tokenizer, feature cache, calibration, candidate pool,
metadata, random seed, dependency versions and inference budget with each result.
Any changed model asset or inference setting requires a new comparison; agreement
for one checkpoint does not imply agreement for another.
