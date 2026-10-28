# ICEBERG

ICEBERG is a forward model: a generator predicts the fragmentation DAG of a molecule and an
intensity model assigns intensities to its fragments. Candidates are ranked by the distance
between the predicted and the measured spectrum. [`__init__.py`](__init__.py) loads the
generator and intensity pair and predicts sparse spectra;
[`models/retrieval/iceberg.py`](../../models/retrieval/iceberg.py) supplies that predictor to
the shared candidate ranking in [`simulators/retrieve.py`](../retrieve.py). FRIGID refinement
and FOAM use the same predictor. `vendor/` contains the generator, intensity predictor, graph
layers, fragmentation and chemistry functions.

Queries need a collision energy in eV, adduct, instrument and precursor m/z. The paper row
uses the frozen Table 3 queries with imputed energies.

## Assets

| Configuration key | Input |
|---|---|
| `generator_checkpoint`, `intensity_checkpoint` | Matching generator and intensity weights (`dag_pred`), both loaded strictly. Table 3 uses the pair retrained on the collision-energy-completed training split (`checkpoints/iceberg/msg_ce_completed_{generator,intensity}.ckpt`); FRIGID and FOAM use the MassSpecGym pair (`checkpoints/iceberg/msg_{generator,intensity}.ckpt`). |
| `candidates` | The complete ordered retrieval candidate pool. |

**Parameters.**

| Key | Default | Meaning |
|---|---|---|
| `query_intensity_space` | `native` | `native` takes the peaks as prepared; `raw` applies one square root to the query intensities |
| `num_bins`, `upper_limit` | 150000, 1500 | spectrum binning: number of bins up to `upper_limit` m/z |
| `sparse_k` | 100 | predicted peaks kept per spectrum |
| `max_heavy_atoms` | 10000 | candidates with more heavy atoms get distance 1 |
| `fragment_threshold` | 0.0 | minimum fragment probability in the generator |
| `max_nodes` | 100 | maximum fragments per molecule |

## Run

From the repository root, after `bash paper/prepare_data.sh`:

```bash
bash paper/table3/iceberg/run.sh
```

The configuration is [`paper/table3/iceberg/config.yaml`](../../../../paper/table3/iceberg/config.yaml).
Checkpoints are listed in the [asset inventory](../../../../docs/assets.md).

## Source

[coleygroup/ms-pred](https://github.com/coleygroup/ms-pred), based on
`e446eebb0f83e53ede016c62522ac2dd371801de`, MIT; see [`vendor/LICENSE.txt`](vendor/LICENSE.txt)
and the [third-party notices](../../../../THIRD_PARTY_NOTICES.md).
