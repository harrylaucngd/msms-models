# msms-models

**Reference implementations behind *MassSpecGym in the Wild*: uncovering and correcting
evaluation pitfalls in AI-driven molecule discovery.**

Every method the paper's authors evaluated in its retrieval (Table 3) and *de novo*
generation (Table 5) leaderboards lives here, under one interface and one fixed
evaluator. Each row of those tables has a folder that reproduces it:

```bash
python -m pip install -e '.[mces]'
bash paper/prepare_data.sh
bash paper/table3/mist_4m_aug/run.sh      # predicts and evaluates
```

See [`paper/`](paper/) for the row-by-row index and inputs.

## Models

| Task | Model | Source | Paper row |
|---|---|---|---|
| Retrieval | `nearest-neighbor` | [models/retrieval/nearest_neighbor](src/msms_models/models/retrieval/nearest_neighbor/) | Table 3 · Nearest Neighbor† |
| | `mist` | [models/retrieval/mist.py](src/msms_models/models/retrieval/mist.py), [encoders/mist](src/msms_models/encoders/mist/) | Table 3 · MIST *4M-aug* |
| | `iceberg` | [simulators/iceberg](src/msms_models/simulators/iceberg/) | Table 3 · ICEBERG |
| | `glacier` | [simulators/glacier](src/msms_models/simulators/glacier/) | Table 3 · GLACIER |
| | `dreams-chemberta` | [models/retrieval/dreams_chemberta](src/msms_models/models/retrieval/dreams_chemberta/) | Table 3 · (1) |
| | `chemformer-two-stage` | [models/retrieval/chemformer_two_stage](src/msms_models/models/retrieval/chemformer_two_stage/) | Table 3 · (2) |
| | `chemberta-classifier` | [models/retrieval/chemberta_classifier](src/msms_models/models/retrieval/chemberta_classifier/) | Table 3 · (3) |
| | `pubchem-order` | [models/retrieval/pubchem_order](src/msms_models/models/retrieval/pubchem_order/) | Table 3 · (4) |
| | `chiral-atoms` | [models/retrieval/chiral_atoms.py](src/msms_models/models/retrieval/chiral_atoms.py) | Table 3 · (5) |
| De novo | `molforge` | [models/de_novo/molforge](src/msms_models/models/de_novo/molforge/) | Table 5 · MIST + MolForge† |
| | `frigid-base`, `frigid` | [models/de_novo/frigid](src/msms_models/models/de_novo/frigid/) | Table 5 · FRIGID-base, FRIGID |
| | `ms-gpt-base` | [models/de_novo/ms_gpt](src/msms_models/models/de_novo/ms_gpt/) | Fig. 2 decoder querying (ScaledMIST) |
| | `msnovelist` | [models/de_novo/msnovelist](src/msms_models/models/de_novo/msnovelist/) | Morgan/MIST adaptation; training and prediction |
| | `foam` | [models/de_novo/foam](src/msms_models/models/de_novo/foam/) | cited in Table 5 |

`msms-models models` lists each model and its task. Methods trained by upstream projects are integrated for inference with
their original weights. The authors' own experiments (the nearest-neighbour index and
the shortcut models) include their training code.

## Usage

```bash
msms-models download msg --output data --candidates formula
msms-models prepare --input data/MassSpecGym.tsv --split test --formula-from-target \
  --output data/msg/queries.jsonl
msms-models prepare-candidates --input data/MassSpecGym_retrieval_candidates_formula.json \
  --queries data/msg/queries.jsonl --output data/msg/candidates_formula.jsonl
msms-models predict  --config paper/table3/mist_4m_aug/config.yaml [--limit 2] [--device cuda:0]
msms-models evaluate --protocol table3 --queries data/msg/queries.jsonl \
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
to the YAML file), and the model's parameters. `predict` runs in the current Python
environment. Models with conflicting dependencies are installed in separate
environments; see [installation](docs/install.md).

## Evaluation

[`evaluation/`](src/msms_models/evaluation/) implements the MassSpecGym metrics:
- **Identity:** InChIKey-14.
- **Similarity:** Morgan r2/2048 Tanimoto without chirality.
- **MCES:** myopic MCES with threshold 15 and the stronger bound. Invalid or missing predictions count as 100.
- **Intervals:** 99.9% BCa over 20,000 resamples.

`--protocol table3|table5` fixes them for each table. Evaluation always keeps every
test spectrum, so missing predictions count against a model. See
[evaluation](docs/evaluation.md).

## Repository layout

```text
src/msms_models/
  models/retrieval/   one module or package per retrieval method
  models/de_novo/     one package per generator
  models/__init__.py  the registry: name, task and module of every model
  encoders/mist/      MIST fingerprint encoders and posterior caches
  simulators/         ICEBERG and GLACIER forward models, spectral retrieval
  data/               versioned downloads, query and candidate records
  evaluation/         metrics and the paper protocols
  api.py, cli.py      predict / run / evaluate
paper/                one folder per reproduced table row
configs/              library configurations beyond the paper (CANOPUS, MS-GPT, FOAM)
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
