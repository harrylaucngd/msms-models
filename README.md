# msms-models

A Python library for molecular structure generation and candidate retrieval from
tandem mass spectra. Methods share data formats, inference interfaces and evaluation
metrics, while retaining their own model architectures and search algorithms.
MassSpecGym and CANOPUS/NPLIB1 are supported datasets.

## Methods

Each method has one source directory. Follow its guide for installation, weights,
input preparation, configurations and inference.

| Method / implementation | Model variants | Task | Datasets |
|---|---|---|---|
| [Nearest Neighbor](src/msms_models/methods/nearest_neighbor/) | `subformula-nn` | Retrieval | MSG |
| [MIST](src/msms_models/methods/mist/) | `mist-noaug`, `mist-4m-aug` | Retrieval; shared fingerprint encoder | MSG retrieval; MSG/CANOPUS encoding |
| [MolForge](src/msms_models/methods/molforge/) | `molforge` | De novo | MSG, CANOPUS |
| [MS-GPT](src/msms_models/methods/ms_gpt/) | `ms-gpt-base`, `ms-gpt` | De novo | MSG, CANOPUS |
| [FRIGID](src/msms_models/methods/frigid/) | `frigid-base`, `frigid` | De novo | MSG, CANOPUS |
| [ICEBERG](src/msms_models/methods/iceberg/) | `iceberg` | Retrieval; shared spectrum predictor | MSG retrieval; MSG/CANOPUS prediction |
| [GLACIER](src/msms_models/methods/glacier/) | `glacier` | Retrieval | MSG |
| [FOAM](src/msms_models/methods/foam/) | `foam` | De novo | MSG, CANOPUS |

`msms-models models` lists the same model variants, required assets, parameters
and configuration locations. Variants share their method's implementation.
Training neural models remains in the linked upstream projects; fitting the
nearest-neighbor index is included here.

## Prepare data

From a checkout, with Python 3.10 or 3.11:

```bash
python -m pip install -e .
msms-models download msg --output data --candidates formula
msms-models prepare --input data/MassSpecGym.tsv --split test --output data/msg/queries.jsonl
msms-models prepare-candidates \
  --input data/MassSpecGym_retrieval_candidates_formula.json \
  --queries data/msg/queries.jsonl --output data/msg/candidates_formula.jsonl
```

The download pins the official dataset revision and verifies SHA-256 checksums.
It includes the spectra TSV and original formula candidate pool (about 604 MB).
Preparation preserves all 17,556 test spectra, supplied formula annotations and
candidate order. Use `--candidates mass` for the mass pool. The corrected candidate
pool is available with `--candidate-version 1.5`; the two versions represent
different evaluation inputs.

See [data preparation](docs/data.md) for training/validation splits, CANOPUS,
feature caches and the input/output schema. For weights, start with your method's
guide and the [asset inventory](docs/assets.md). **MIST's advertised asset uploads
are still pending**, so a fresh download-only neural workflow is not yet available
for every configuration. Missing downloads are identified explicitly.

## Infer and evaluate

Install the [optional dependencies](docs/install.md) for the chosen method and set
asset paths in its [configuration](docs/configuration.md). For MIST retrieval:

```bash
msms-models doctor --config configs/mist/msg-4m-aug.yaml
msms-models infer --config configs/mist/msg-4m-aug.yaml --device cuda:0
msms-models evaluate --task retrieval --queries data/msg/queries.jsonl \
  --predictions results/mist-4m-aug/msg/predictions.jsonl \
  --output results/mist-4m-aug/msg/metrics.json
```

Paths in a configuration are relative to that file. The default device is CPU;
select a GPU explicitly. Each run records its configuration, input hashes,
dependency versions, ordered predictions and method diagnostics. `doctor` checks
inputs and import availability; actual inference checks checkpoint compatibility.
Use `smoke --config ... --output results/smoke/example --limit 2` to exercise the
same inference and evaluation on a small subset.

Evaluation always uses the original query manifest. Missing and unsupported
predictions remain in the denominator. Install `.[mces]` and add `--mces` for
bounded MCES distances. See [evaluation](docs/evaluation.md) for molecular identity,
fingerprints, candidate handling and confidence intervals.

The same interface is available in Python:

```python
from msms_models import RunConfig, infer, evaluate, MetricProtocol
from msms_models.io import load_queries, read_records

config = RunConfig.from_file("configs/molforge/msg.yaml")
infer(config)
summary, per_spectrum = evaluate(
    load_queries(config.queries, require_target=True),
    read_records(config.output / "predictions.jsonl"),
    MetricProtocol(task="de-novo", cutoffs=(1, 10)),
)
```

For in-process use, `predict(config, query_records)` invokes the same method and
returns prediction records. `infer(config)` manages files and an isolated Python
worker, useful when model dependencies need separate environments.

## Structure and development

```text
src/msms_models/
  methods/<method>/  model loading, inference, specification, source and guide
  datasets/          versioned downloads and query/candidate preparation
  retrieval/         shared fingerprint and candidate operations
  posteriors.py      cached fingerprint validation and spectrum-ID alignment
  simulation.py      shared spectral retrieval and metadata checks
  metrics.py         common molecule evaluation
  api.py             Python prediction and isolated inference interfaces
  cli.py             command-line access to those same interfaces
configs/<method>/    dataset configurations, with named variants where needed
tests/               behavioral, protocol and integration tests
```

Read [architecture](docs/architecture.md) to add a method and [validation](docs/validation.md)
for testing. The [MSG in the Wild results note](docs/paper-results.md) maps Tables 3
and 5 to method implementations, configurations and cited results. It also records
which experimental assets still need release. Paper runs use the regular interfaces.

## Attribution

Original integration code is MIT licensed. Included third-party components retain
their licenses, including **CC BY-NC-SA 4.0 for MolForge** and Apache-2.0 for
MS-GPT/FRIGID components. [Third-party notices](THIRD_PARTY_NOTICES.md) identify
source revisions and modifications. Weights and datasets have their own licenses.
Please cite the [method papers and upstream projects](docs/references.md).
