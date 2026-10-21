# Installation and environments

The core package requires Python 3.10 or newer, NumPy, SciPy, pandas, PyYAML and
RDKit. It can prepare manifests and evaluate predictions without importing a
neural model. Retrieval from an already computed MIST posterior also uses only
the core dependencies. Install from a checkout with `python -m pip install -e .`.

Use separate environments for model families when dependency versions differ.
Set `python` in the run YAML to the desired interpreter. Omit it to use the
interpreter running `msms-models`. Each inference call starts a new process;
model namespaces and graph-library versions do not leak into other runs.

| Backend | Package extras and additional requirements |
|---|---|
| Sub-formula NN | `[nn]`; use NumPy 1.24.4, SciPy 1.10.1 and RDKit 2023.9.4 for the official NN implementation |
| MIST retrieval | `[mist]` |
| MolForge | `[mist,molforge]` |
| MS-GPT-base | `[ms-gpt-base]`; includes its historical MIST architecture |
| MS-GPT | `[ms-gpt]`; includes PEFT for the required adapter |
| FRIGID-base | `[mist,frigid]` |
| ICEBERG | `[iceberg]` and matching graph-library wheels |
| FRIGID refinement | `[frigid,iceberg]` and matching graph-library wheels; prepare MIST posteriors in the generation environment |
| FOAM | `[foam,iceberg]` and matching graph-library wheels |
| GLACIER | `[mist,iceberg,glacier]` and matching graph-library wheels |

For example, in a model environment:

```bash
python -m pip install -c environments/generation.txt -e '.[ms-gpt]'
```

MS-GPT uses the official SAFE tokenizer assets (SAFE-Mol 0.1.13); full MS-GPT additionally
uses PEFT 0.19.1. The decoder architecture lives beside its run configuration.

The [generation constraints](../environments/generation.txt) record the Linux
CPython 3.10 combination used for real GPU inference: PyTorch 2.9.1,
Transformers 4.49.0 and NumPy 1.26.4. SAFE-Mol 0.1.13 cannot import with
Transformers 4.52.4; MS-GPT's decoder uses the equivalent per-layer KV-cache API
of the pinned 4.49 release. Tests compare cached generation with full-prefix
generation.

FRIGID uses Lightning 2.5.1, SAFE-Mol 0.1.13 and BioNeMo-MoCo 0.0.2.1.
BioNeMo requires NumPy 1; GLACIER's public ms-pred dependency requires NumPy 2.
Use separate environments. Preserve the length predictor's feature order and
its serialization dependencies (NGBoost 0.5.8, scikit-learn 1.7.2, joblib 1.5.2).
The original paper length predictors remain unavailable; runtime validation used
a separately fitted predictor and does not establish their numerical equivalence.
The SAFE tokenizer is always an explicit local asset.

The graph backends use DGL 1.1.3, torch-scatter 2.1.2 and PyTorch Lightning 1.9.5.
Use setuptools 77–80 for the legacy Lightning runtime; the graph extra includes
this constraint. The exact NN reference extra targets Python 3.10/3.11.
Install DGL and PyTorch Geometric extension wheels compatible with your PyTorch,
CUDA and operating-system versions before installing the graph extra. Consult
[PyG installation](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
and [DGL installation](https://www.dgl.ai/pages/start.html). A CPU-only setup is
supported. Avoid installing a different DGL major version without revalidation.

## Legacy graph environment

For ICEBERG, FOAM and FRIGID refinement on Linux, use Python 3.10 and
[these constraints](../environments/legacy-graph.txt):

```bash
python -m venv .venv-graph
.venv-graph/bin/python -m pip install --upgrade pip
.venv-graph/bin/python -m pip install 'torch==2.6.0'
.venv-graph/bin/python -m pip install torch-scatter \
  -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
.venv-graph/bin/python -m pip install -c environments/legacy-graph.txt -e '.[frigid,iceberg,foam]'
```

This PyTorch release uses CUDA 12.4 wheels; choose matching CPU wheels if needed.
The tested DGL wheel is CPU-only, so these graph checks use `device: cpu`.
Set the configuration's `python` to `.venv-graph/bin/python` using an absolute
path. Prepare MIST posteriors in the generation environment and supply their
probabilities, IDs and manifest to FRIGID. Do not install every extra together.
The PyArrow 19 pin avoids a native import crash observed with PyArrow 25 in this
PyTorch 2.6 environment. An import-only `doctor` check cannot detect this crash;
use the documented inference smoke command after installation.

## Public GLACIER dependency

GLACIER uses the public MIST compatibility architecture and the following pinned
public helper implementation:

```bash
python -m pip install -c environments/glacier.txt -e '.[mist,iceberg,glacier]'
```

The `glacier` extra pins `ms-pred` to commit
`708148c2a8eb0c32120644436fefd2fe90cd2214`; a sibling checkout is unnecessary.
The GLACIER weights linked by coleygroup/MIST
include layers absent from the standard upstream model at this revision;
`mist.glacier` provides their matching architecture with strict loading.
[The MIST compatibility notes](https://github.com/coleygroup/MIST/blob/main/docs/glacier.md)
explain the boundary.

Use a separate environment with [GLACIER constraints](../environments/glacier.txt).
Install the torch-scatter wheel matching PyTorch 2.7/CUDA 12.6 first. On the
validation server, prebuilt PyG 2.7 and 2.9 wheels required a newer glibc than the
host. Building the CPU extension locally resolved that incompatibility:

```bash
CUDA_VISIBLE_DEVICES="" FORCE_CUDA=0 MAX_JOBS=2 python -m pip install \
  --no-build-isolation --no-deps --no-binary torch-scatter \
  --force-reinstall torch-scatter==2.1.2
```

This needs a C++ compiler and an already installed PyTorch. It supports CPU
validation, not CUDA scatter operations. If torchvision is present, pair it
with PyTorch (0.22.0 for PyTorch 2.7); a leftover version from another environment
can fail during Lightning imports.

ICEBERG, FOAM and FRIGID refinement use the included legacy ICEBERG inference
namespace. They do not require installing a second `ms-pred` checkout. This
closure retains the feature vocabulary of the released weights, including the
MSG/CANOPUS distinction in instrument embeddings. It does not expose upstream
training entry points.

## Verification

```bash
python -m pip install -e '.[test]'
python -m pytest
msms-models --help
msms-models doctor --config configs/molforge/msg.yaml
```

`doctor` reports file presence and dependency availability. It does not certify
that a checkpoint matches an architecture: the model loader checks that during
inference. MCES additionally needs `[mces]` and an available CBC executable,
which the evaluator checks before computing any distances.

## Native CBC on macOS

PuLP may bundle an Intel CBC executable that cannot execute on Apple Silicon.
Use a native CBC installation on PATH (for example `brew install cbc`) and select
it explicitly:

```bash
msms-models evaluate --task de-novo --queries data/msg/queries.jsonl \
  --predictions results/molforge/msg/predictions.jsonl \
  --output results/molforge/msg/metrics.json --mces --mces-solver COIN_CMD
```

The Python equivalent is `MetricProtocol(mces=True, mces_solver="COIN_CMD")`.
The selected solver is part of the recorded protocol and MCES cache key. File
presence alone does not prove that a binary can execute on the current platform.
