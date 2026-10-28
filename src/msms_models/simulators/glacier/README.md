# GLACIER

GLACIER is a forward model: it predicts a spectrum for each candidate molecule, and
candidates are ranked by the distance between that prediction and the measured spectrum.
[`__init__.py`](__init__.py) loads the checkpoint and predicts sparse spectra;
[`models/retrieval/glacier.py`](../../models/retrieval/glacier.py) supplies that predictor to
the shared candidate ranking in [`simulators/retrieve.py`](../retrieve.py). The architecture is
`mist.glacier` from the pinned MIST package, with helpers from the pinned ms-pred package
(see [installation](../../../../docs/install.md)); the GLACIER checkpoint has layers that the
standard ms-pred architecture lacks.

Queries need a collision energy in eV, adduct, instrument and precursor m/z. The paper row
uses the frozen Table 3 queries with imputed energies.

## Assets

| Configuration key | Input |
|---|---|
| `checkpoint` | GLACIER weights for the `mist.glacier` architecture (the paper row uses `checkpoints/glacier/msg_ce_completed.ckpt`). |
| `candidates` | The complete ordered retrieval candidate pool. |

**Parameters.**

| Key | Default | Meaning |
|---|---|---|
| `query_intensity_space` | `native` | `native` takes the peaks as prepared; `raw` applies one square root to the query intensities |
| `num_bins`, `upper_limit` | 150000, 1500 | spectrum binning: number of bins up to `upper_limit` m/z |
| `sparse_k` | 100 | predicted peaks kept per spectrum |
| `max_heavy_atoms` | 10000 | candidates with more heavy atoms get distance 1 |

## Run

From the repository root, after `bash paper/prepare_data.sh`:

```bash
bash paper/table3/glacier/run.sh
```

The configuration is [`paper/table3/glacier/config.yaml`](../../../../paper/table3/glacier/config.yaml).
Checkpoints are listed in the [asset inventory](../../../../docs/assets.md).

## Source

The architecture is in [coleygroup/MIST](https://github.com/coleygroup/MIST) at
`4d09208fceaa9a34579493a70095416405b76563`; the helpers come from
[coleygroup/ms-pred](https://github.com/coleygroup/ms-pred) at
`708148c2a8eb0c32120644436fefd2fe90cd2214` (MIT). Both are installed as dependencies; see the
[third-party notices](../../../../THIRD_PARTY_NOTICES.md).
