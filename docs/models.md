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
for example, MIST vMassSpecGym and MIST *4M-aug* are two checkpoints of the `mist` model.
`frigid-base` and `frigid` are two entries because they need different assets.

## Run configurations

```yaml
model: molforge
dataset: msg
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

Every configuration lives in its paper entry's folder under [`paper/`](../paper/).

MIST-conditioned models read fingerprint posteriors in one of three forms:
- a posterior cache (`posterior`, `posterior_ids`);
- a ScaledMIST checkpoint with its features (`mist_checkpoint`, `compact_cache`);
- a MistNet checkpoint with subformula trees (`mist_checkpoint`, `diffms_features`):
  `mist_architecture: diffms` for the checkpoint released with DiffMS, `massspecgym` for
  MIST vMassSpecGym.

For MistNet, `attention_padding: masked` excludes padding from attention;
`additive-bool` adds the boolean padding mask to the attention scores, which makes the
posteriors depend on the batch (paper §6.1.1).

## Running

| Command | Effect |
|---|---|
| `msms-models predict --config C` | writes `predictions.jsonl` and `queries.jsonl` |
| `msms-models fit-nn --train T --validation V --output DIR` | builds the nearest-neighbour index |

## Where each method lives

```text
models/retrieval/random.py           MassSpecGym random ranking (optionally formula-filtered)
models/retrieval/deepsets.py, fingerprint_ffn.py   MassSpecGym fingerprint-predicting baselines
models/retrieval/nearest_neighbor/   sub-formula neighbour consensus (index, fitting, vendor features)
models/retrieval/mist.py             fingerprint ranking with MIST posteriors
models/retrieval/iceberg.py, glacier.py   forward-model retrieval (simulators/ owns the models)
models/retrieval/dreams_chemberta/   DreaMS + ChemBERTa alignment (SpecBridge)
models/retrieval/chemformer_two_stage/  Chemformer two-stage generative retrieval (GLMR)
models/retrieval/chemberta_classifier/  spectrum-blind SMILES classifier
models/retrieval/pubchem_order/      spectrum-blind PubChem ordering
models/retrieval/chiral_atoms.py     spectrum-blind stereocentre count
models/de_novo/{random_generation,smiles_transformer}/   MassSpecGym de novo baselines
models/de_novo/{molforge,ms_gpt,frigid,foam,msnovelist}/
massspecgym/                         featurization, layers and training loop of the MassSpecGym baselines
encoders/mist/                       ScaledMIST, MistNet (DiffMS and MassSpecGym releases), posterior caches
simulators/{iceberg,glacier}/        forward spectrum models shared by retrieval, FRIGID and FOAM
```

Each package has a `README.md` covering its assets, its source and how to run it; the
single-file MassSpecGym baselines are described in their paper entry folders and module
docstrings. Methods with their own training run it as `python -m <module>` (the
MassSpecGym retrieval baselines), `python -m <package>.train` or `.fit`, or, for the
nearest-neighbour index, `msms-models fit-nn`.

## Adding a model

1. Write `predict(config, queries)` under `models/<task>/`. Read inputs with `config.require_assets` and `data.candidates.load_candidates`.
2. Add a `Model(...)` entry to the registry.
3. Add a configuration, its dependencies to an extra in `pyproject.toml` and to [`environments/lock.sh`](../environments/lock.sh), and a guide listing its assets and parameters.
