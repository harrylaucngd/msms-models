# FRIGID

`inference.py` implements checkpoint loading and adapters for one-shot generation and refinement. `vendor/` contains the DLM, sampler, masking and formula/length utilities. Refinement scores candidates with the [ICEBERG simulator](../../../simulators/iceberg/README.md), so it reads the same query fields (adduct, instrument, precursor m/z and collision energy).

`frigid-base` is the first scratch round. `num_rounds` counts **all** rounds: ten refinement transitions require `num_rounds: 11`. The CANOPUS configuration uses 24 refinements (`num_rounds: 25`). A dataset-specific length model and its ordered feature list are needed.

## Assets

| Configuration key | Input |
|---|---|
| `decoder_checkpoint` | DLM weights from the official checkpoint archive. |
| `safe_tokenizer` | The SAFE tokenizer paired with that decoder. |
| `length_model, length_features` | Dataset-specific serialized length predictor and its ordered feature list. |
| `generator_checkpoint, intensity_checkpoint` | Matching ICEBERG pair; required by refinement, unused by `frigid-base`. |

Fingerprint conditioning uses `mist_checkpoint` and `compact_cache`. Alternatively, supply `posterior` and `posterior_ids` to reuse a cache without loading MIST. The [data guide](../../../../../docs/data.md#mist-posteriors) defines these formats.

**Parameters.**

| Key | Default | Meaning |
|---|---|---|
| `fingerprint_threshold` | none | posterior threshold for the conditioning fingerprint |
| `num_samples` | 128 | proposals per spectrum (`frigid-base`) |
| `batch_size` | 128 | proposals per round (`frigid`) |
| `num_rounds` | 11 | all rounds, the first included |
| `num_unique_to_refine` | 8 | best unique candidates re-masked each round |
| `masks_per_molecule` | 12 | masked copies of each refined candidate |
| `mask_probability` | 0.75 | token masking probability |
| `hallucinated_peaks` | 10 | predicted peaks absent from the measured spectrum that guide masking |
| `temperature`, `randomness` | 1.0, 10.0 | sampler softmax temperature and randomness |
| `sigma_lambda` | 3.0 | variance multiplier for the NGBoost length samples |
| `simulation_energies`, `simulation_nce` | 10–90 step 10, `true` | simulated energies; `simulation_nce` reads them as normalized and converts to eV |
| `max_nodes`, `sparse_k` | 100, 100 | ICEBERG fragments per molecule and peaks kept |
| `measured_collision_energy` | `recorded` | `unknown` ignores the query energies of the measured spectra |

The encoder keys are in the [MIST guide](../../../encoders/mist/README.md).

Provider links and current download availability are in the [asset inventory](../../../../../docs/assets.md). Paths are explicit in the configurations below.

## Run

From the repository root, after the [common data setup](../../../../../docs/data.md) and [asset preparation](../../../../../docs/assets.md):

```bash
python -m pip install -e '.[mist,frigid,iceberg]'
msms-models predict --config paper/table5/frigid_base/config.yaml --limit 2 --output results/smoke/frigid-base
msms-models predict --config paper/table5/frigid_base/config.yaml
msms-models evaluate --protocol table5 --queries results/paper/table5/frigid_base/queries.jsonl \
  --predictions results/paper/table5/frigid_base/predictions.jsonl \
  --output results/paper/table5/frigid_base/metrics.json
```

`--limit 2` exercises the configured inference on the first two queries; it is a plumbing check, not a benchmark result.

## Variants and source

| Model ID | Configurations | Entry point |
|---|---|---|
| `frigid-base` | [msg](../../../../../paper/table5/frigid_base/config.yaml), [canopus](../../../../../configs/frigid/canopus-base.yaml) | [inference.py](inference.py) |
| `frigid` | [msg](../../../../../paper/table5/frigid/config.yaml), [canopus](../../../../../configs/frigid/canopus.yaml) | [inference.py](inference.py) |

Shared loading, fingerprint ranking and evaluation remain in the library core instead of being copied per method. [Third-party notices](../../../../../THIRD_PARTY_NOTICES.md) record the source revisions and licenses.
