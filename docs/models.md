# Model protocols

Model selection and hyperparameter fitting must use training/validation data.
The test query manifest is a frozen evaluation universe. Default inference
settings are visible in the [configurations](../configs); changed budgets and
thresholds must be reported with the results.

## De novo

**MIST + MolForge** thresholds a 4,096-bit MIST posterior, encodes sorted on-bit
indices with the source SentencePiece model, and decodes with the original
six-layer transformer and beam search. Both tokenizers, the decoder and the
posterior threshold are explicit assets/settings. The official decoder's positional
encoding, source length 104 and target length 130 are retained. Formula filtering
is an explicit option, disabled by default. A beam-search numerical agreement
test does not establish agreement for a different tokenizer or decoder family.

**MS-GPT-base** and **MS-GPT** share the conditional molecule-language model,
training-calibrated posterior density band, and 20-query × 5-sample default budget.
Base uses the pretrained decoder alone; full MS-GPT additionally loads a
posterior-aligned LoRA adapter trained before inference on training spectra.
The adapter is required for `ms-gpt` and forbidden for `ms-gpt-base`.
This distinction follows the paper's ablation; a single-threshold point query is
a separate ablation, not MS-GPT-base.

The decoder and density-query functions follow the
[official MS-GPT implementation](https://github.com/VIKI623/MS-GPT).
Use its released SAFE tokenizer. Candidate structures are grouped by connectivity,
filtered by formula and ranked by frequency with band-max Tanimoto tie-breaking,
as in the official `scripts/rank_candidates.py`. Querying and ranking use strict
`>` thresholding. Match the encoder, density calibration, tokenizer, decoder and
adapter to the same training recipe.

**FRIGID-base** uses known formula and thresholded MIST fingerprints in a SAFE
discrete diffusion model. Formula-only length prediction uses explicit model and
feature-list assets. Base inference is exactly the first native refinement round: 128 scratch
proposals by default, with no spectral prediction. Valid unique structures follow
the native formula-priority and fingerprint-similarity ordering. It does not keep
sampling until a requested number of formula matches has been found.
All learned weights must load. Historical shared-attention aliases are removed
only after verifying tensor equality; only deterministic derived buffers may
be reconstructed.

**FRIGID** additionally runs the native spectral-error-guided refinement loop.
The first round generates `B` molecules. Later rounds combine `B−M×K` fresh
proposals with up to `M×K` remasked proposals from the top `K` structures. ICEBERG
predicts spectra for the candidates used by the next round; hallucinated peaks
guide fragment/token masking. The MSG configuration uses eleven total rounds (ten refinement transitions), `B=128`, `K=8`,
`M=12`, masking probability 0.75, and simulated normalized energies 10–90.
Measured collision energies remain in eV. The final unused forward prediction
is skipped. The refinement loop follows the [official FRIGID sampler](https://github.com/coleygroup/FRIGID).

**FOAM** retains formula-constrained crossover/mutation, nondominated selection
and candidate-fingerprint crowding from the native graph optimizer. Its two
objectives are native spectral entropy similarity and `1−SA/10`. Final output is
ordered by spectral similarity, as in upstream reporting. It consumes explicit
initial seeds, uses no target structure in scoring, and runs no external candidate
search. The default MSG recipe uses 1,500 optimization calls, population 200,
offspring 600 and one island. Seed evaluations are accounted for separately.
Its native spectral bins are 15,000 over 0–1,500 Da; these differ from the
150,000-bin forward-retrieval recipe. Population plots and target-dependent
logging are omitted; benchmark metrics are computed by the shared evaluator.

## Retrieval

**Sub-formula-only NN** annotates peaks using the known precursor formula, builds
a sparse token space from training spectra, and combines token similarity with
a soft molecular-formula similarity term. It does **not** require an exact
formula match in the training library. An unseen formula is still compared with
all training formulas; no test spectrum is dropped. Query tokens outside the
training vocabulary are removed before normalization. The selected neighbors
produce a continuous consensus fingerprint, which ranks the supplied molecular
candidates by Tanimoto similarity.

`fit-nn` uses only training spectra to construct the library and only validation
spectra to select `w`, `alpha`, temperature and neighbor count. The generated index records the selected parameters and full
selection protocol. Test inference does no tuning. Preserve the reference NumPy
version when exact neighbor identity is required at an unstable `argpartition`
tie. Candidate ranking itself uses stable ties by default.

**MIST no augmentation / 4M augmentation** use distinct released checkpoints.
No augmentation means no added synthetic training spectra; it does not assert
that ordinary peak/intensity perturbation was disabled during training. The 4M
label refers to synthetic molecules, not the number of generated spectra.
Retrieval uses continuous 4,096-bit posterior/candidate Tanimoto by default.
Binary fingerprint ranking and its threshold must be requested explicitly.

**ICEBERG / GLACIER** simulate candidate spectra using the declared query
metadata and rank the complete supplied candidate pool by native entropy distance.
The default retrieval grid has 150,000 bins over 0–1,500 Da. ICEBERG keeps the top
100 unbinned fragment peaks and sums bins; GLACIER keeps the top 100 binned peaks
with max pooling. Experimental prepared peaks use max pooling. Native exact-key
and connectivity-key expansion restores candidate aliases. Unsupported candidates
retain their places with distance 1; unexpected runtime failures abort the run.
Ties retain input candidate order. Both legacy ICEBERG instrument vocabularies
and the GLACIER compatibility architecture are checked by strict weight loading.
