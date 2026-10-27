# Data

## MassSpecGym

```bash
msms-models download msg --output data --candidates formula
msms-models prepare --input data/MassSpecGym.tsv --split test --formula-from-target \
  --output data/msg/queries.jsonl
msms-models prepare-candidates --input data/MassSpecGym_retrieval_candidates_formula.json \
  --queries data/msg/queries.jsonl --output data/msg/candidates_formula.jsonl
```

`download` fetches the [official dataset](https://huggingface.co/datasets/roman-bushuiev/MassSpecGym)
at the revision pinned in [`data/massspecgym.json`](../src/msms_models/data/massspecgym.json).
- `--candidates mass|both` selects the other pools; Table 2 uses the mass-based one.
- `--molecules` adds the 4M MassSpecGym molecules (MCES-disjoint from the test fold) on which the
  SMILES Transformer's BPE tokenizer is trained.
- `--candidate-version 1.5` selects the RDKit-canonicalized pools, keyed by the canonical target SMILES. Results on different pool versions are not comparable.
  - `prepare-candidates` finds a target's 1.5 list by canonicalizing it. Newer RDKit releases write 4 of the 3,170 test molecules differently from the release that built the pools; those are matched by the first InChIKey block, which gives the same lists as RDKit 2025.9.

`prepare` keeps every spectrum of the split in source order. `--formula-from-target`
declares the *bonus* (formula-known) challenge: it fills `formula` from the target and,
for the common adducts, `precursor_formula`. [`paper/prepare_data.sh`](../paper/prepare_data.sh) prepares the
train, validation and test inputs that the paper rows use.

> **Candidate order.** Both official pools list the true molecule first. A ranker
> that breaks ties by the supplied order can therefore find the answer without
> reading the spectrum. The spectrum-blind baselines break ties randomly for this
> reason, and random retrieval draws a score for every candidate.

## Records

**Queries.** A JSONL file with one spectrum per line:

```json
{"id": "q1", "formula": "C2H6O", "precursor_formula": "C2H7O", "precursor_mz": 47.0491,
 "adduct": "[M+H]+", "instrument": "Orbitrap", "collision_energy": 20.0,
 "spectrum": {"mz": [29.0391, 31.0184], "intensity": [0.3, 1.0]}, "target_smiles": "CCO"}
```

- IDs are unique, and peaks are sorted by m/z.
- `target_smiles` is needed only for evaluation and fitting, and inference never reads it.
- `prepare` also reads CSV/TSV/JSON and the annotated parquet exports released with MS-GPT. MassSpecGym column names (`identifier`, `smiles`, `mzs`, `intensities`, `instrument_type`, `fold`) are recognized.

**Candidates.** Ordered, per query: `{"id": "q1", "candidates": ["CCO", "COC"]}`.
- An official JSON keyed by target SMILES is also accepted as input to `prepare-candidates`.
- Duplicates and order are preserved.

**Predictions.** `{"id": "q1", "candidates": [...], "scores": [...], "status": "ok"}`, with
the best candidate first. A spectrum that a model cannot handle keeps its row, with an
empty list and an explanatory `status`.

## Inputs some models need

| Input | Models | Notes |
|---|---|---|
| `formula`, `precursor_formula` | NN, MIST, decoders, random generation, SMILES Transformer | from `--formula-from-target` for the bonus challenge |
| `precursor_mz`, peaks | DeepSets, SMILES Transformer | the precursor is the first token (`SpecTokenizer`) |
| `adduct`, `instrument`, `precursor_mz`, `collision_energy` (eV) | ICEBERG, GLACIER, FOAM | missing energies are never inferred from the target |
| measured peaks | FRIGID refinement | raw peaks; ms-pred's `process_spec_file` is applied, and an unknown energy is allowed |
| MIST posteriors or features | MIST, MolForge, MS-GPT, FRIGID | see below |

**Collision energies.** About half of the MassSpecGym test spectra have no collision
energy. The paper's ICEBERG and GLACIER rows used the energies imputed in Appendix A.5.
They are released as a frozen query file; see [paper/README.md](../paper/README.md#inputs-and-assets).

**Forward-model inputs.** ICEBERG and GLACIER take `query_intensity_space`:
- `native`: the peaks already have the model's prepared intensities.
- `raw`: one square root is applied.

## MIST posteriors

MIST-conditioned models accept either of two inputs:
- **Encoder inputs:** a ScaledMIST checkpoint with its feature cache (`mist_checkpoint`, `compact_cache`). The vDiffMS diagnostic rows instead use DiffMS's MIST (`diffms_features`).
- **A posterior cache:** two files:
  - `posterior`: an `N × 4096` float array in [0, 1] (Morgan radius 2, no chirality);
  - `posterior_ids`: the matching spectrum IDs.

Rows are aligned by ID. A fingerprint threshold belongs to the encoder that produced
the posteriors, so calibrate a new one for every new encoder. Running a MIST-conditioned
model from a checkpoint writes its posteriors to `<output>/posterior/`.

## CANOPUS / NPLIB1

`dataset` in a run configuration is a label (`msg` or `canopus`); only the DiffMS MIST
encoder reads it. CANOPUS configurations for FOAM, FRIGID and MolForge are under
[`configs/`](../configs/) and point at a local export. Prepare it with
`msms-models prepare --input <export> --output data/canopus/queries.jsonl`. The public
MS-GPT NPLIB1 export has 672 test spectra, while the original CANOPUS experiments used
819, so keep the original query file when comparing with them.
