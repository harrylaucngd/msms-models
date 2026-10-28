# FRIGID

FRIGID generates molecules with a SAFE diffusion language model (DLM) conditioned on a MIST
fingerprint and the query's formula, with the sequence length sampled from a dataset-specific
NGBoost model of the formula. `frigid-base` is one round of generation from scratch. `frigid` adds
refinement rounds: candidates are scored with the [ICEBERG simulator](../../../simulators/iceberg/README.md),
and the tokens of the best unique candidates are re-masked, guided by predicted peaks absent
from the measured spectrum, and sampled again. `num_rounds` counts all rounds, so ten
refinement rounds are `num_rounds: 11`. [`inference.py`](inference.py) loads the checkpoint
and runs both modes; `vendor/` contains the DLM, sampler, masking and formula and length
utilities. Refinement reads the same query fields as ICEBERG (adduct, instrument, precursor
m/z and collision energy).

## Assets

| Configuration key | Input |
|---|---|
| `decoder_checkpoint` | DLM weights from the official checkpoint archive. |
| `safe_tokenizer` | The SAFE tokenizer paired with that decoder. |
| `length_model`, `length_features` | Dataset-specific serialized length predictor and its ordered feature list. |
| `generator_checkpoint`, `intensity_checkpoint` | ICEBERG pair; required by refinement, unused by `frigid-base`. |
| `posterior`, `posterior_ids` | MIST posteriors (the paper rows use MIST *4M-aug*); the encoder keys are in the [MIST guide](../../../encoders/mist/README.md). |

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

## Run

From the repository root, after `bash paper/prepare_data.sh`:

```bash
bash paper/table5/frigid_base/run.sh   # frigid-base
bash paper/table5/frigid/run.sh        # frigid, ten refinement rounds
```

The configurations are [`paper/table5/frigid_base/config.yaml`](../../../../../paper/table5/frigid_base/config.yaml)
and [`paper/table5/frigid/config.yaml`](../../../../../paper/table5/frigid/config.yaml).
Checkpoints are listed in the [asset inventory](../../../../../docs/assets.md).

## Source

[coleygroup/FRIGID](https://github.com/coleygroup/FRIGID), based on
`d242a2c06d671e0fb6fbc1a5b31f0e19e24b456d`, Apache-2.0; see
[`vendor/LICENSE.txt`](vendor/LICENSE.txt) and the
[third-party notices](../../../../../THIRD_PARTY_NOTICES.md).
