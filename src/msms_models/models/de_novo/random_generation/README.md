# Random chemical generation (MassSpecGym)

The random de novo baseline of MassSpecGym (Table 5: Random generation). For each query's
formula it enumerates valence assignments over common valences, keeps up to 10 feasible ones,
and builds random molecular graphs from them until 10 molecules exist. With element
statistics, each new bond's neighbour atom and bond order are sampled from how often they
occur in the training molecules. No spectrum is read.

[`vendor/random_generator.py`](vendor/random_generator.py) is MassSpecGym's
`massspecgym/models/de_novo/random.py` at `485deb22df` (unchanged at `f259fe3780`) with only
the Lightning plumbing removed. [`fit.py`](fit.py) records the statistics as MassSpecGym's one
training epoch did, and [`inference.py`](inference.py) generates for each query in order,
seeding Python's `random` and NumPy as `pytorch_lightning.seed_everything` did.

**Behaviour kept from upstream.**
- Molecules are returned with explicit hydrogens and are never sanitized: upstream calls
  `Chem.RemoveHs` but discards its result. Parsing a returned SMILES can add implicit
  hydrogens, and a molecule can be disconnected when no spanning tree was found.
- With `cache_results`, a formula seen before returns the same molecules.
- The statistics count every training spectrum, so molecules with several spectra weigh more.
- Their dictionaries are filled in the order the molecules arrive, and part of that order comes
  from Python sets of strings. Sampling depends on that order, not on its distribution, so a
  given seed reproduces molecules exactly only with the same statistics file.

**Paper settings.** MassSpecGym's evaluation notebook names the formula-known run
`random_baseline_formula` (MCES 21.11 / 18.25, Tanimoto 0.08 / 0.11, the cited row). Its code
is not in the repository. The mass-based run in `notebooks/demo.ipynb`, which reproduces that
challenge's printed numbers, uses `max_top_k=10`, `estimate_chem_element_stats=True`,
`enforce_connectivity=False` and default `count_of_valid_valence_assignments=10`, after one
training epoch and `seed_everything(0)`. The formula-known run is taken to be the same with
`formula_known=True`.

## Assets

| Configuration key | Input |
|---|---|
| `element_stats` | the statistics written by `fit.py` (needed with `estimate_chem_element_stats`) |

**Parameters.** Defaults are `RandomDeNovo`'s. `estimate_chem_element_stats` (false) samples
bonds from the statistics, `enforce_connectivity` (true) first grows a random spanning tree,
`count_of_valid_valence_assignments` (10), `max_top_k` (10) and `cache_results` (true). The
configuration's `seed` seeds generation.

## Run

```bash
bash environments/install.sh   # from the repository root
python -m msms_models.models.de_novo.random_generation.fit \
  --train data/msg/train.jsonl --output checkpoints/random_generation/element_stats.pkl
msms-models predict --config paper/table5/random_generation/config.yaml
```

## Source

[pluskal-lab/MassSpecGym](https://github.com/pluskal-lab/MassSpecGym) (Bushuiev et al.,
NeurIPS 2024), MIT; see [`vendor/LICENSE.txt`](vendor/LICENSE.txt) and the
[third-party notices](../../../../../THIRD_PARTY_NOTICES.md).
