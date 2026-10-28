# Random chemical generation (MassSpecGym)

The random de novo baseline of MassSpecGym (Table 5: Random generation). For each query's
formula it enumerates valence assignments over common valences, keeps up to 10 feasible ones,
and builds random molecular graphs from them until 10 molecules exist. With element
statistics, each new bond's neighbour atom and bond order are sampled from how often they
occur in the training molecules. No spectrum is read.

[`vendor/random_generator.py`](vendor/random_generator.py) is MassSpecGym's generator
(`massspecgym/models/de_novo/random.py`) without the Lightning module code. [`fit.py`](fit.py)
counts the element and bond statistics over every training spectrum, so molecules with
several spectra weigh more. [`inference.py`](inference.py) generates for each query in
order, seeding Python's `random` and NumPy with the configuration's `seed`. Molecules are
returned with explicit hydrogens and are not sanitized, and a molecule can be disconnected
when no spanning tree is found. With `cache_results`, a formula seen before returns the same
molecules.

## Assets

| Configuration key | Input |
|---|---|
| `element_stats` | the statistics written by `fit.py` (needed with `estimate_chem_element_stats`) |

**Parameters.** `estimate_chem_element_stats` (false) samples bonds from the statistics,
`enforce_connectivity` (true) first grows a random spanning tree,
`count_of_valid_valence_assignments` (10), `max_top_k` (10) and `cache_results` (true). The
paper row sets `estimate_chem_element_stats: true`, `enforce_connectivity: false` and
`seed: 0`.

## Run

From the repository root, after `bash paper/prepare_data.sh`:

```bash
bash paper/table5/random_generation/run.sh
```

`run.sh` fits the statistics first if they are missing:

```bash
python -m msms_models.models.de_novo.random_generation.fit \
  --train data/msg/train.jsonl --output checkpoints/random_generation/element_stats.pkl
```

## Source

[pluskal-lab/MassSpecGym](https://github.com/pluskal-lab/MassSpecGym) at `485deb22df`
(Bushuiev et al., NeurIPS 2024), MIT; see [`vendor/LICENSE.txt`](vendor/LICENSE.txt) and the
[third-party notices](../../../../../THIRD_PARTY_NOTICES.md).
