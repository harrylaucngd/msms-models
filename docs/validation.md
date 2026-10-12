# Validation

Validate three separate properties: implementation agreement with the official
method, compatibility of the selected weights and inputs, and benchmark accuracy.
Passing an inference smoke test does not establish full-test accuracy.

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
