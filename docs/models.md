# Model protocols

Model selection and hyperparameter fitting must use training/validation data.
The test query manifest is a frozen evaluation universe. Default inference
settings are visible in the [configurations](../configs); changed budgets and
thresholds must be reported with the results.

## De novo

**DiffMS** loads the complete trained encoder, fingerprint projection and graph
decoder state. Its model JSON records decoder dimensions, encoder architecture,
noise schedule, training graph statistics and atom ordering. The included model
JSONs describe the scaled-MIST checkpoint family; a standalone pretrained
fingerprint-to-graph decoder is not a full spectrum-to-structure checkpoint.
Input heavy-atom counts come from the known formula; hydrogens are implicit and
no target bonds are supplied. The native decoder constrains heavy-atom composition,
not the final hydrogen count; there is no additional formula post-filter.
The current supported atom vocabulary is C/O/P/N/S/Cl/F/H. Unsupported formulas
produce empty predictions. Sampling uses 500 native diffusion steps and ranks
valid sampled InChIs by frequency. Atom order, sample batching and seed affect
individual draws and are recorded.

**MIST + MolForge** thresholds a 4,096-bit MIST posterior, encodes sorted on-bit
indices with the source SentencePiece model, and decodes with the original
six-layer transformer and beam search. Both tokenizers, the decoder and the
posterior threshold are explicit assets/settings. The historical positional
encoding, source length 104 and target length 130 are retained. Formula filtering
is an explicit option, disabled by default. A beam-search numerical agreement
test does not establish agreement for a different tokenizer or decoder family.

**MS-GPT** uses the SMILES character decoder family, with its 98-token vocabulary,
and a training-calibrated posterior density band. Default inference forms
20 binary queries and generates five samples from each. Bits use a strict `>`
threshold. Candidate structures are grouped by connectivity identity, filtered
by known formula and ordered by frequency; ties use the maximum fingerprint
similarity over 20 points in the density band. Calibration files identify the
benchmark and training split. Use the band associated with the exact MIST encoder;
legacy calibration values are not interchangeable with a new encoder's values.

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
guide fragment/token masking. The default recipe uses ten rounds, `B=128`, `K=8`,
`M=12`, masking probability 0.75, and simulated normalized energies 10–90.
Measured collision energies remain in eV. The final unused forward prediction
is skipped. This backend really performs refinement; increasing a base sampler's
sample count alone is a different procedure.

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
spectra to select `w`, `alpha`, temperature and neighbor count. The reference MSG
selection is `w=2`, `alpha=2`, `T=0.1`, `k=50`; the released index records the full
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
