# Architecture

A method is the unit of implementation and maintenance. Dataset selection,
checkpoints and inference budgets configure that method. Evaluation operates on
its predictions through the same contract for every method.

## Method ownership

Each `src/msms_models/methods/<method>/` directory owns:

- `spec.py`: model variants, supported tasks/datasets, configuration locations,
  required assets, accepted parameters and input validation. It imports no neural
  runtime when the registry is read.
- `inference.py`: `predict(config, queries)`, returning ordered prediction records.
  Model construction and native generation/ranking behavior belong here or in
  explicitly named modules in the same directory.
- Model source: architecture, decoder or optimization modules. Bundled upstream
  source is under `vendor/` with its license and revision in the third-party notices.
- `README.md`: source map, dependencies, assets and normal usage.

Variants are specifications over a shared implementation. For example,
`frigid-base` and `frigid` share a decoder, while the latter adds refinement.
`configs/frigid/msg-base.yaml` and `configs/frigid/msg.yaml` configure those variants
on MSG. No source code is copied for each dataset or experiment.

The registry gathers lightweight specifications. `get_model(name)` exposes the
selected specification. The Python and command-line APIs dispatch through its
explicit implementation path; adding a method does not require another branch
inside the worker or shared evaluator.

## Shared components and dependencies

`datasets` prepares versioned inputs. `io` validates spectrum records and aligns
predictions by ID. `retrieval.fingerprints` ranks the supplied candidate universe.
`simulation` handles spectral candidate retrieval through a supplied prediction
function; it does not select a neural architecture by method name. `metrics`
evaluates an ordered prediction set against a fixed query universe.

Cross-method model dependencies have a single owner:

- `methods/mist/encoder.py` runs ScaledMIST; `methods/mist/legacy.py` runs the
  historical benchmark-paired architecture with annotated spectrum inputs.
  Retrieval and fingerprint-conditioned
  generators use it through the posterior interface. Cached posteriors bypass
  MIST loading and are checked and aligned by `posteriors.py`.
- `methods/iceberg/inference.py` loads the ICEBERG generator/intensity pair and
  predicts spectra. ICEBERG retrieval, FOAM and FRIGID refinement reuse that
  implementation.
- `methods/glacier/inference.py` owns GLACIER loading and its distinct spectral
  pooling procedure. The shared retrieval algorithm receives this predictor.

The pinned MIST package supplies its neural architecture and GLACIER compatibility
architecture; their exact installation is documented. Local bundled methods require
no sibling checkout, mutable `sys.path`, or duplicated dependencies. Historical
`backends` and private `_vendor` paths are compatibility imports, not source owners.

## Public interfaces

`predict(config, queries)` runs in the current interpreter, validates the method's
input contract, returns ordered predictions, and writes method diagnostics to
`config.output`. It is also the call made inside the isolated worker.

`infer(config)` loads a query file, records inputs and configuration, starts the
configured interpreter, verifies output IDs, and writes the run record. The CLI
calls this API. A smoke run uses a smaller query manifest and this same path; it
does not replace a model with a stand-in.

`evaluate(queries, predictions, protocol)` returns aggregate and per-spectrum
metrics. Its computation does not depend on a method ID, paper title or reference
number. Missing predictions remain in the denominator; invalid candidate slots
remain in their original rank positions under the default policy.

Inference never derives formula inputs from target molecules. Explicit benchmark
preparation can construct known-formula inputs. Candidate pools preserve supplied
strings, duplicates and tie order. Posterior rows align by spectrum ID.

## Assets and configurations

`configs/<method>/` contains dataset configurations and variants. These are the
normal entry points for benchmark runs as well as new data. Configuration paths
are relative to their YAML file. Known checkpoint hashes can be pinned using
`asset_sha256`; mismatches fail before inference.

Runs record file hashes and a source fingerprint. For directories, the recorder
hashes manifest files and explicitly reports that it has not hashed every file.
Full cache integrity requires a versioned, content-verified cache release.

The paper-to-method mapping is documentation. Reported reference numbers, table
row names and citation-only methods are not part of package execution or model
registration. The runtime contains only implemented capabilities.

## Add a method

1. Implement `predict(config, queries)` in its method directory; keep optional
   neural imports inside runtime functions.
2. Declare its variants, requirements and configuration paths in `spec.py`, then
   add that specification module to `registry.METHODS`.
3. Add configurations under `configs/<method>/` and document source, assets,
   environment and scientific assumptions in the method guide.
4. Reuse common preparation and evaluation. Put method-specific validation in
   the method specification, and shared contract checks in the common validator.
5. Test ID alignment, invalid/unsupported inputs, ranking, checkpoint loading and
   agreement with the relevant upstream implementation. Exercise real assets
   through `infer` or `smoke` in the intended environment.
