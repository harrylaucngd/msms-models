# FOAM

FOAM is a formula-constrained graph genetic algorithm: starting from seed structures, it
evolves molecules of the query's formula and scores each by the entropy similarity between
the measured spectrum and the spectrum [ICEBERG](../../../simulators/iceberg/README.md)
predicts for it. [`inference.py`](inference.py) runs the optimizer in `vendor/` (crossover,
mutation, selection and the entropy objective) on each query and returns every visited
molecule, ranked by its entropy similarity.
The seeds are an explicit input, supplied independently of the target.

## Assets

| Configuration key | Input |
|---|---|
| `generator_checkpoint`, `intensity_checkpoint` | ICEBERG generator and intensity weights for spectral scoring (`checkpoints/iceberg/msg_{generator,intensity}.ckpt`). |
| `seeds` | ID-keyed initial structures for each query. |

**Parameters.**

| Key | Default | Meaning |
|---|---|---|
| `max_calls` | 1500 | objective evaluations per spectrum |
| `population_size`, `offspring_size` | 200, 600 | genetic-algorithm population and offspring per generation |
| `starting_seed_size` | -1 | seeds used to start; -1 uses all |
| `keep_population` | 20000 | size of the candidate buffer |
| `num_islands`, `mutation_rate` | 1, 0.2 | island count and mutation rate |
| `truncate` | `true` | truncate the population each generation |
| `patience`, `log_frequency` | 10, 100 | early-stopping patience and logging interval |
| `num_bins`, `upper_limit` | 15000, 1500 | spectrum binning for the entropy objective |
| `ignore_precursor_peak` | `true` | leave the precursor bin out of the entropy distance |
| `max_nodes` | 100 | ICEBERG fragments per molecule |

## Run

From the repository root, after `bash paper/prepare_data.sh`, build the seeds (DiffMS
predictions for the test spectra plus the PubChem structures with the query's formula) with
[`paper/table5/foam/build_seeds.py`](../../../../../paper/table5/foam/build_seeds.py), then
run the row:

```bash
python paper/table5/foam/build_seeds.py --queries data/msg/queries.jsonl \
  --diffms <diffms predictions.jsonl> --pubchem CID-SMILES.gz \
  --output data/paper/table5/foam/seeds.jsonl
bash paper/table5/foam/run.sh
```

The configuration is [`paper/table5/foam/config.yaml`](../../../../../paper/table5/foam/config.yaml);
the [row README](../../../../../paper/table5/foam/README.md) describes the seed inputs.

## Source

[coleygroup/foam](https://github.com/coleygroup/foam) at
`f3ed69b11642c01be0301dd2cfcba5cb3ee8c793`, MIT; see [`vendor/LICENSE.txt`](vendor/LICENSE.txt)
and the [third-party notices](../../../../../THIRD_PARTY_NOTICES.md).
