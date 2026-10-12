# msms-models

Unified inference and evaluation for molecular structure generation and candidate
retrieval from tandem mass spectra. Inputs, model assets and ranked predictions
use shared file formats; each model retains its native decoding or search method.
Training remains in the upstream projects.

| Task | Models |
|---|---|
| De novo, known molecular formula | DiffMS, MIST + MolForge, MS-GPT, FRIGID-base, FRIGID with spectral refinement, FOAM |
| MassSpecGym candidate retrieval | Sub-formula nearest neighbor, MIST no augmentation, MIST 4M augmentation, ICEBERG, GLACIER |

De novo adapters accept MassSpecGym (`msg`) and CANOPUS/NPLIB1 (`canopus`) assets.
Retrieval recipes cover the standard MassSpecGym test split and both candidate
pools. A formula candidate pool itself supplies formula information; the NN and
MIST encoders also require known formulas. See [model protocols](docs/models.md)
for the assumptions and supported checkpoint families.

## Start

```bash
python -m pip install -e .
msms-models models
msms-models prepare --input data/MassSpecGym.tsv --split test --output data/queries.jsonl
```

Install the optional dependencies for your model using the
[installation guide](docs/install.md), obtain the assets described in
[the asset guide](docs/assets.md), then edit a [configuration](configs/):

```bash
msms-models doctor --config configs/mist-noaug.yaml
msms-models infer --config configs/mist-noaug.yaml --dry-run
msms-models infer --config configs/mist-noaug.yaml
msms-models evaluate --task retrieval --queries data/queries.jsonl \
  --predictions results/mist-noaug/predictions.jsonl \
  --output results/mist-noaug/metrics.json
```

`doctor` checks required files and import availability; inference checks actual
checkpoint compatibility. Each run executes in its configured Python environment
and produces predictions, an input manifest, a protocol record, dependency
versions and a log.
Paths in a YAML file are relative to that file. CPU inference is the default;
GPU use and CPU thread limits are explicit configuration fields.

## Data and evaluation

Use the [input contracts](docs/data.md) to prepare spectrum inputs, candidate
libraries, MIST features and fingerprint posterior caches. Released compact MIST
caches are suitable for inference; target fingerprint arrays are not read.

The evaluator requires an ordered query manifest. Missing predictions and
unsupported spectra remain in the denominator. It reports connectivity identity
through InChIKey14 and structure Tanimoto with 2,048-bit radius-2 Morgan
fingerprints. Confidence intervals use spectrum-level BCa bootstrapping with
20,000 resamples and 99.9% confidence. MCES is optional:

```bash
python -m pip install -e '.[mces]'
msms-models evaluate --task de-novo --queries data/queries.jsonl \
  --predictions results/diffms/predictions.jsonl --output results/diffms/metrics.json \
  --mces --mces-cache results/mces.sqlite --mces-threads 1
```

See the [validation scope](docs/validation.md) for checkpoint and regression checks.
See [evaluation details](docs/evaluation.md) for invalid predictions, duplicate
candidates, tie handling, MCES limits and normalization. Model-native candidate
postprocessing happens before evaluation and is recorded separately.

## Implementation and attribution

The package includes the required decoder, optimization, nearest-neighbor and
legacy ICEBERG inference components in separate namespaces. It uses the public
[coleygroup/MIST](https://github.com/coleygroup/MIST) package for spectrum encoding
and GLACIER checkpoint compatibility. GLACIER additionally uses the pinned public
[coleygroup/ms-pred](https://github.com/coleygroup/ms-pred) helpers. There are no
required sibling repositories or Git submodules.

MIST is based on Goldman et al.,
[Annotating metabolite mass spectra with domain-inspired chemical formula transformers](https://doi.org/10.1038/s42256-023-00708-3).
The public MIST implementation supports modern distributed training and optional
synthetic spectra; `mist-noaug` and `mist-4m-aug` select distinct trained assets.
Please also cite the papers for the methods you use, listed in
[references and upstream training](docs/references.md).

Original integration code is MIT licensed. Included third-party components retain
their licenses, including **CC BY-NC-SA 4.0 for the MolForge decoder** and
Apache-2.0 for MS-GPT/FRIGID components. See [third-party notices](THIRD_PARTY_NOTICES.md).
Model weights and datasets are separate downloads with their own licenses.
