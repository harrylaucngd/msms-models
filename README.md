# msms-models

**Reference implementations behind *MassSpecGym in the Wild*: uncovering and correcting
evaluation pitfalls in AI-driven molecule discovery.**

Every method entry of the paper's main text that is not cited from its authors is
implemented here, under one interface, one environment and one fixed evaluator. Each
entry of Tables 1–5 and Figure 2 has a folder that reproduces it:

```bash
bash environments/install.sh && source .venv/bin/activate   # see docs/install.md
bash paper/prepare_data.sh
hf download harrylau/msms-models --local-dir . --include 'checkpoints/*' 'data/*'
bash paper/table3/mist_4m_aug/run.sh      # predicts and evaluates
```

See [`paper/`](paper/) for the entry-by-entry index, the cited entries and the inputs.

## Models

| Task | Model | Source | Paper entries |
|---|---|---|---|
| Retrieval | `random` | [models/retrieval/random.py](src/msms_models/models/retrieval/random.py) | Table 2 · Random (± ground-truth formula); Table 3 · Random |
| | `deepsets` | [models/retrieval/deepsets.py](src/msms_models/models/retrieval/deepsets.py), [massspecgym](src/msms_models/massspecgym/) | Table 3 · DeepSets (+ Fourier features) |
| | `fingerprint-ffn` | [models/retrieval/fingerprint_ffn.py](src/msms_models/models/retrieval/fingerprint_ffn.py), [massspecgym](src/msms_models/massspecgym/) | Table 3 · Fingerprint FFN |
| | `nearest-neighbor` | [models/retrieval/nearest_neighbor](src/msms_models/models/retrieval/nearest_neighbor/) | Table 3 · Nearest Neighbor† |
| | `mist` | [models/retrieval/mist.py](src/msms_models/models/retrieval/mist.py), [encoders/mist](src/msms_models/encoders/mist/) | Table 1 · settings 1–5; Table 2 · MIST; Table 3 · MIST vMassSpecGym, MIST *4M-aug* |
| | `iceberg` | [simulators/iceberg](src/msms_models/simulators/iceberg/) | Table 3 · ICEBERG |
| | `glacier` | [simulators/glacier](src/msms_models/simulators/glacier/) | Table 3 · GLACIER |
| | `dreams-chemberta` | [models/retrieval/dreams_chemberta](src/msms_models/models/retrieval/dreams_chemberta/) | Table 3 · (1) |
| | `chemformer-two-stage` | [models/retrieval/chemformer_two_stage](src/msms_models/models/retrieval/chemformer_two_stage/) | Table 3 · (2) |
| | `chemberta-classifier` | [models/retrieval/chemberta_classifier](src/msms_models/models/retrieval/chemberta_classifier/) | Table 3 · (3) |
| | `pubchem-order` | [models/retrieval/pubchem_order](src/msms_models/models/retrieval/pubchem_order/) | Table 3 · (4) |
| | `chiral-atoms` | [models/retrieval/chiral_atoms.py](src/msms_models/models/retrieval/chiral_atoms.py) | Table 3 · (5) |
| De novo | `random-generation` | [models/de_novo/random_generation](src/msms_models/models/de_novo/random_generation/) | Table 5 · Random generation |
| | `smiles-transformer` | [models/de_novo/smiles_transformer](src/msms_models/models/de_novo/smiles_transformer/) | Table 5 · SMILES, SELFIES Transformer |
| | `msnovelist` | [models/de_novo/msnovelist](src/msms_models/models/de_novo/msnovelist/) | Table 5 · MIST + MSNovelist |
| | `foam` | [models/de_novo/foam](src/msms_models/models/de_novo/foam/) | Table 5 · FOAM |
| | `molforge` | [models/de_novo/molforge](src/msms_models/models/de_novo/molforge/) | Tables 4–5 · MIST + MolForge† (MIST 4M-aug, MIST vDiffMS) |
| | `frigid-base`, `frigid` | [models/de_novo/frigid](src/msms_models/models/de_novo/frigid/) | Table 5 · FRIGID-base, FRIGID |
| | `ms-gpt-base` | [models/de_novo/ms_gpt](src/msms_models/models/de_novo/ms_gpt/) | Figure 2 · decoder querying |

`msms-models models` lists each model and its task. Models without released weights
include their training code, which the entry's `run.sh` or README runs.

## Usage

```bash
msms-models download msg --output data --candidates formula
msms-models prepare --input data/MassSpecGym.tsv --split test --formula-from-target \
  --output data/msg/queries.jsonl
msms-models prepare-candidates --input data/MassSpecGym_retrieval_candidates_formula.json \
  --queries data/msg/queries.jsonl --output data/msg/candidates_formula.jsonl
msms-models predict  --config paper/table3/mist_4m_aug/config.yaml [--device cuda:0]
msms-models evaluate --protocol table3 --queries results/paper/table3/mist_4m_aug/queries.jsonl \
  --predictions results/paper/table3/mist_4m_aug/predictions.jsonl --output metrics.json
```

The same interface is available from Python:

```python
from msms_models import RunConfig, predict, evaluate
from msms_models.data.io import load_queries
from msms_models.evaluation.protocols import get_protocol

config = RunConfig.from_file("paper/table5/mist_molforge/config.yaml")
queries = load_queries(config.queries)
summary, per_spectrum = evaluate(queries, predict(config, queries), get_protocol("table5"))
```

A run configuration names a model, its dataset, its inputs and assets (paths relative
to the YAML file), and the model's parameters. Every model runs in the one locked
environment; see [installation](docs/install.md).

## Evaluation

[`evaluation/`](src/msms_models/evaluation/) implements the MassSpecGym metrics:
- **Identity:** InChIKey-14.
- **Similarity:** Morgan r2/2048 Tanimoto without chirality.
- **MCES:** myopic MCES with threshold 15 and the stronger bound. Invalid or missing predictions count as 100.
- **Intervals:** 99.9% BCa over 20,000 resamples.

`--protocol table2|table3|table5` fixes them for each table, and `evaluate-fingerprints`
gives the encoder Tanimoto of Tables 1 and 4. Evaluation always keeps every
test spectrum, so missing predictions count against a model. See
[evaluation](docs/evaluation.md).

## Repository layout

```text
src/msms_models/
  models/retrieval/   one module or package per retrieval method
  models/de_novo/     one package per generator
  models/__init__.py  the registry: name, task and module of every model
  massspecgym/        featurization, layers and training of the MassSpecGym baselines
  encoders/mist/      MIST fingerprint encoders and posterior caches
  simulators/         ICEBERG and GLACIER forward models, spectral retrieval
  data/               versioned downloads, query and candidate records
  evaluation/         metrics and the paper protocols
  api.py, cli.py      predict / run / evaluate
paper/                one folder per paper entry: configurations, run.sh, README
docs/                 install, data, evaluation, models, assets
```

Each method's source lives in one place, together with a guide. Unmodified
third-party code sits in that method's `vendor/` folder with its license; see the
[third-party notices](THIRD_PARTY_NOTICES.md). To add a model, see
[models](docs/models.md).

## Development

```bash
python -m pip install -e '.[test,mces]'
ruff check src tests && python -m pytest
```

## License and citation

The integration and evaluation code is MIT. Included third-party components keep
their licenses: **CC BY-NC-SA 4.0 for MolForge**, Apache-2.0 for MS-GPT, FRIGID and
Chemformer, and MIT for the others. Weights and datasets carry their own terms.
Please cite the paper and the [original methods](docs/references.md).

```bibtex
@article{liu2026massspecgym,
  title   = {MassSpecGym in the Wild: Uncovering and Correcting Evaluation Pitfalls in AI-Driven Molecule Discovery},
  author  = {Liu, Hongxuan and Bushuiev, Roman and Lightheart, Ivy and Manjrekar, Mrunali and Bushuiev, Anton and
             Lederbauer, Magdalena and Jozefov, Filip and Wang, Yinkai and Hassoun, Soha and Sivic, Josef and
             Taylor, James and Wang, Runzhong and Healey, David and Pluskal, Tom{\'a}{\v{s}} and Coley, Connor W.},
  year    = {2026},
}
```
