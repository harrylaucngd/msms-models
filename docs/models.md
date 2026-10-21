# Models and configurations

## The registry

[`models/__init__.py`](../src/msms_models/models/__init__.py) has one `Model` entry per
model. Reading it imports no model runtime:

```python
Model(
    "molforge", "de-novo", "msms_models.models.de_novo.molforge.inference",
    datasets=("msg", "canopus"),
    assets=("decoder_checkpoint", "source_tokenizer", "target_tokenizer"),
    parameters=("beam_size", "formula_filter", ...),
    requires=("torch", "sentencepiece"),
    uses_mist=True,
)
```

The module named there implements two functions:
- `predict(config, queries)` returns one prediction record per query, in query order.
- `validate(config, queries)`, optionally, checks the method-specific inputs before any weights are loaded.

Heavy imports stay inside functions. A model's variants are configurations, not code:
for example, MIST *noaug* and *4M-aug* are two checkpoints of the `mist` model.
`frigid-base` and `frigid` are two entries because they need different assets.

## Run configurations

```yaml
model: molforge
dataset: msg                 # msg | canopus
queries: ../../../data/msg/queries.jsonl
output: ../../../results/paper/table5/mist_molforge
device: cpu                  # or cuda:0; override with --device
workers: 1
seed: 42
assets:                      # files and directories, relative to this YAML file
  decoder_checkpoint: ../../../checkpoints/molforge/decoder.pth
  posterior: ...
parameters:                  # only the model's declared parameters are accepted
  fingerprint_threshold: 0.18791014701128006
asset_sha256:                # optional: fail before inference if a file differs
  decoder_checkpoint: 67710e7b...
```

MIST-conditioned models (`uses_mist`) take one of three input sets:
- a posterior cache (`posterior`, `posterior_ids`, `posterior_manifest`);
- a ScaledMIST checkpoint with its features (`mist_checkpoint`, `compact_cache`);
- the MS-GPT-released architecture (`mist_checkpoint`, `legacy_features`).

For the legacy encoder, `attention_padding: masked` is the default, which is the
padding fix of paper §6.1.1. `additive-bool` restores the historical batch-dependent
behaviour for diagnostics only.

## Running

| Command | Effect |
|---|---|
| `msms-models doctor --config C` | checks inputs, assets, hashes and runtime imports, without loading weights |
| `msms-models predict --config C [--limit N]` | writes `predictions.jsonl`, `queries.jsonl`, `inputs.json` (hashes), `environment.json` and `run.json` |
| `msms-models fit-nn --train T --validation V --output DIR` | builds the nearest-neighbour index |

`predict` refuses to overwrite a directory that already holds predictions.
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
encoders/mist/                       ScaledMIST, MS-GPT legacy MIST, DiffMS MIST, posterior caches
simulators/{iceberg,glacier}/        forward spectrum models shared by retrieval, FRIGID and FOAM
```

Each package has a `README.md` covering its assets, its source and how to run it.

## Adding a model

1. Write `predict(config, queries)` (and `validate` if needed) under `models/<task>/`. Read inputs with `config.require_assets` and `data.candidates.load_candidates`.
2. Add a `Model(...)` entry to the registry with its assets, parameters and imports.
3. Add a configuration, an optional dependency extra in `pyproject.toml`, and a guide.
4. Test the input contract, ID alignment and ranking. Compare outputs with the upstream implementation on real assets, and record what was compared in the guide.
