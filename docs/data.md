# Input and output contracts

## Query manifests

A JSONL file contains one object per spectrum, in evaluation order:

```json
{"id":"example-001","formula":"C2H6O","precursor_formula":"C2H7O","precursor_mz":47.04914,"adduct":"[M+H]+","instrument":"Orbitrap","collision_energy":20.0,"spectrum":{"mz":[29.0391,31.0184],"intensity":[0.3,1.0]}}
```

This is a format example, not a benchmark spectrum. `target_smiles` is optional
for inference and required for evaluation. `formula` is a declared known input;
the inference adapters do not derive it from target structures. Spectrum IDs
must be nonempty and unique. Masses must be positive, intensities nonnegative,
and all numeric values finite. Peaks are sorted by mass during normalization.

`prepare` reads CSV, TSV, JSON and JSONL. Official MassSpecGym column aliases
(`identifier`, `smiles`, `mzs`, `intensities`, `instrument_type`, `fold`) are
recognized. `--split test` retains the official test rows in source order.
For a dataset without a formula column, `--formula-from-target` explicitly
creates a **formula-known benchmark**; this is an input assumption, not formula
prediction. Existing inconsistent formulas are rejected.

Spectrum-based NN needs neutral `formula`, ion `precursor_formula`, and peaks.
MIST-conditioned methods need the matching compact feature rows or an explicit
posterior cache. Forward retrieval, FOAM and FRIGID refinement also need adduct,
instrument, precursor mass and collision energy in **eV**. Missing collision
energies are not inferred from target structures. To reproduce a benchmark that
uses imputed collision energies, supply its frozen metadata and retain that
provenance alongside the input manifest.

Forward methods require `query_intensity_space`. `native` means the supplied
peaks already have the method's prepared intensities. `raw` applies one square
root and is supported for standalone ICEBERG/GLACIER retrieval. It does not
reproduce the paper's peak assignment/filtering from an arbitrary raw spectrum.
FOAM and FRIGID refinement require prepared `native` inputs.

## Candidate pools and optimization seeds

Retrieval candidates are ordered and query-specific:

```json
{"id":"example-001","candidates":["CCO","COC"]}
```

A JSON mapping from query ID to a list is also accepted. Official benchmark JSON
files keyed by target SMILES are supported as a compatibility input, but an
ID-keyed file is preferable for target-free deployment. Candidate order is the
tie-break order. Duplicates remain in the retrieval universe.

Use separate files for the standard mass-based and bonus formula-based pools;
run and report them separately. Formula-aware encoders remain formula-known even
when searching the mass-based pool. The package does not silently reduce a pool
to whatever candidates a forward model successfully predicted.

FOAM uses the same list format for **initial seeds**, plus a JSON seed manifest:

```json
{"dataset":"msg","sources":["provided generator predictions","frozen PubChem formula candidate export"],"uses_target_structures":false}
```

The caller provides those seed sources explicitly. There is no automatic PubChem
lookup or addition of the query's target molecule. Seed evaluations are outside
the native FOAM optimization-call budget; report the initial seed count too.

## MIST compact features and posteriors

Download the prepared compact MSG/CANOPUS cache matching the checkpoint. Each
split contains a manifest, relative part paths and feature arrays. Inference
reads formulas, adducts, intensities, peak types, lengths, instruments and row
names; it does not read target fingerprints or structure IDs. Exact row names
are preferred; a unique acquisition suffix `_collision …` can be aligned to the
original spectrum ID. Ambiguous or missing matches fail.

To construct a cache from processed spectra and formula assignments, use the
public MIST tools:

```bash
mist-build-real-cache --dataset-name msg_all --data-dir data --output data/compact/msg
```

See [MIST data preparation](https://github.com/coleygroup/MIST/blob/main/docs/data.md)
for the source HDF5 layout. A compact cache is a feature representation, not a
new dataset split. Use the released split selection for exact reproduction.

An externally computed posterior consists of three assets:

- `posterior`: an `N × 4096` NumPy probability array, finite and in `[0, 1]`.
- `posterior_ids`: an `N`-element Unicode NumPy ID array.
- `posterior_manifest`: JSON with `dataset` and
  `fingerprint: "morgan-r2-4096-achiral"`, plus the encoder/checkpoint provenance.

Rows are aligned by ID. MolForge, MS-GPT, FRIGID and MIST retrieval can consume
these arrays. Match calibration thresholds to the encoder that produced them.

## Predictions

Each JSONL row contains `id`, an ordered `candidates` list and `status`.
Optional `scores` are aligned with candidates; higher values rank first.
Retrieval uses fingerprint similarity or negative spectral distance. Generators
use their native scores/counts where available. Failed or unsupported spectra
have an empty candidate list and an explanatory status; they are not dropped.
