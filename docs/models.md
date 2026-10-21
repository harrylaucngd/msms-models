# Models and configurations

## The registry

[`models/__init__.py`](../src/msms_models/models/__init__.py) has one `Model` entry per
model. Reading it imports no model runtime:

```python
Model("molforge", "de-novo", "msms_models.models.de_novo.molforge.inference")
```

The module named there implements `predict(config, queries)`, which returns one
prediction record per query. Its assets and parameters are listed in its guide.

Heavy imports stay inside functions. A model's variants are configurations, not code:
for example, MIST *noaug* and *4M-aug* are two checkpoints of the `mist` model.
`frigid-base` and `frigid` are two entries because they need different assets.

## Run configurations

```yaml
model: molforge
dataset: msg                 # a label: msg | canopus
queries: ../../../data/msg/queries.jsonl
output: ../../../results/paper/table5/mist_molforge
device: cpu                  # or cuda:0; override with --device
workers: 1
seed: 42
assets:                      # files and directories, relative to this YAML file
  decoder_checkpoint: ../../../checkpoints/molforge/decoder.pth
  posterior: ...
parameters:
  fingerprint_threshold: 0.18791014701128006
```

`dataset` is a label; only the DiffMS MIST encoder reads it. CANOPUS configurations are
under [`configs/`](../configs/) and point at a local export (see [data](data.md#canopus--nplib1)).

MIST-conditioned models read ScaledMIST fingerprints in one of two forms:
- a posterior cache (`posterior`, `posterior_ids`);
- a ScaledMIST checkpoint with its features (`mist_checkpoint`, `compact_cache`).

The vDiffMS diagnostic rows alone use DiffMS's MIST (`mist_checkpoint`, `diffms_features`).
The MIST architecture released with MS-GPT and FRIGID is deliberately not supported:
its posteriors carry the problems described in the paper's §6.1.

For the DiffMS encoder, `attention_padding: masked` is the default, which is the
padding fix of paper §6.1.1. `additive-bool` restores the historical batch-dependent
behaviour for the diagnostic rows only.

## Running

| Command | Effect |
|---|---|
| `msms-models predict --config C [--limit N]` | writes `predictions.jsonl` and `queries.jsonl` |
| `msms-models fit-nn --train T --validation V --output DIR` | builds the nearest-neighbour index |

`--limit` runs the configured budget on the first N queries; it is a smoke test, not
a benchmark result.

## Where each method lives

```text
models/retrieval/nearest_neighbor/   sub-formula neighbour consensus (index, fitting, vendor features)
models/retrieval/mist.py             fingerprint ranking with MIST posteriors
models/retrieval/iceberg.py, glacier.py   forward-model retrieval (simulators/ owns the models)
models/retrieval/dreams_chemberta/   DreaMS + ChemBERTa alignment (SpecBridge)
models/retrieval/chemformer_two_stage/  Chemformer two-stage generative retrieval (GLMR)
models/retrieval/chemberta_classifier/  spectrum-blind SMILES classifier
models/retrieval/pubchem_order/      spectrum-blind PubChem ordering
models/retrieval/chiral_atoms.py     spectrum-blind stereocentre count
models/de_novo/{molforge,ms_gpt,frigid,foam}/
encoders/mist/                       ScaledMIST, DiffMS MIST (diagnostics), posterior caches
simulators/{iceberg,glacier}/        forward spectrum models shared by retrieval, FRIGID and FOAM
```

Each package has a `README.md` covering its assets, its source and how to run it.

## Adding a model

1. Write `predict(config, queries)` under `models/<task>/`. Read inputs with `config.require_assets` and `data.candidates.load_candidates`.
2. Add a `Model(...)` entry to the registry.
3. Add a configuration, an optional dependency extra in `pyproject.toml`, and a guide listing its assets and parameters.
