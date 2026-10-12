# Validation scope

Validation distinguishes numerical agreement, runnable checkpoint support and
full benchmark evaluation. A small inference test does not establish a model's
full-test accuracy or reproduce results for every possible checkpoint.

| Component | Acceptance evidence |
|---|---|
| Sub-formula NN | 32 standard-test queries against all 194,119 training spectra: identical neighbors and consensus arrays to the reference run; portable index save/load preserved outputs |
| MIST | MSG and CANOPUS, no synthetic augmentation and augmentation: each real checkpoint tested on two measured spectra; spectrum-only encoding exactly matched public MIST, including reversed ID alignment |
| MolForge | Real decoder and both tokenizers strictly loaded; beam candidates and scores matched the original decoder on fixed inputs |
| MS-GPT | Real SMILES decoder strictly loaded; generated tokens matched the native implementation at a fixed seed; formula/frequency ranking checked |
| FRIGID | Real DLM weights and explicit tokenizer/length predictor loaded; shared attention aliases verified; native generation comparison and a two-round spectral refinement fixture completed |
| DiffMS | Full MSG/CANOPUS checkpoints strictly loaded; encoder outputs matched native code on measured spectra; 500-step diffusion completed using target-free heavy-atom formula graphs |
| ICEBERG | Real MSG/CANOPUS generator/intensity pairs strictly loaded with their respective instrument vocabularies; candidate retrieval completed |
| GLACIER | Real checkpoint loading and candidate retrieval completed; sparse entropy scoring matched the native function |
| FOAM | Real ICEBERG models used in a synthetic spectrum/seed fixture; optimization evaluated newly generated structures in addition to initial seeds |
| Installed package | All eleven inference entries completed through the isolated worker API using the installed wheel |

The core package was also installed in a new environment and tested without the
source directory on its import path. Neural checks used the documented dependency
stack. Optional graph/CUDA dependencies are not exercised by the small CPU CI
suite; use the matching environment and weights for those backends.

## Full-cohort metric regression

The shared evaluator was checked against frozen sub-formula NN retrieval outputs
for **all 17,556 standard test spectra**, separately for mass and formula candidate
pools. Per-spectrum Hit@1/5/20 and top-1 Tanimoto agreed with the reference. Cached,
already-solved MCES values produced the same mean and 99.9% BCa interval. Those
MCES pairs were not re-solved for this aggregation check. Separate real CBC checks
verified the solver call and persistent-cache reload on small molecule pairs.

One regression test covers explicit hydrogens retained for double-bond
stereochemistry. Rewriting such inputs through stereochemistry removal and a
SMILES round-trip changed fingerprints in 119 mass-pool queries. The evaluator
therefore computes the benchmark's achiral fingerprint directly on the input
molecular graph, while retaining the separate prescribed MCES normalization.

These checks validate the implementation and evaluation contract. They do not
claim a new complete benchmark run for every included model. Exact reproduction
also requires the matching checkpoint, feature cache, calibration, candidate
pool, metadata, random seed and inference budget.
