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
| Sub-formula NN | `[nn]`; use NumPy 1.24.4, SciPy 1.10.1 and RDKit 2023.9.4 for the frozen reference index |
| MIST retrieval | `[mist]` |
| MolForge | `[mist,molforge]` |
| MS-GPT | `[mist,ms-gpt]` |
| DiffMS | `[mist,diffms]` |
| FRIGID-base | `[mist,frigid]` |
| ICEBERG | `[iceberg]` and matching graph-library wheels |
| FRIGID refinement | `[mist,frigid,iceberg]` and matching graph-library wheels |
| FOAM | `[foam,iceberg]` and matching graph-library wheels |
| GLACIER | `[mist,iceberg]`, matching graph-library wheels and public `ms-pred` below |

For example, in a model environment:

```bash
python -m pip install -e '.[mist,ms-gpt]'
```

The tested neural stack uses PyTorch 2.7.0, Transformers 4.52.4, NumPy 2.2.6,
SciPy 1.15.3 and RDKit 2026.3.1. FRIGID uses Lightning 2.5.1, SAFE-Mol 0.1.13 and
BioNeMo-MoCo 0.0.2.1; its released length predictors were checked with NGBoost
0.5.8, scikit-learn 1.7.2 and joblib 1.5.2. Preserve the serialized length model's
feature order. The SAFE tokenizer is an explicit local asset; inference does
not fetch an arbitrary tokenizer revision from the network.

The graph backends use DGL 1.1.3, torch-scatter 2.1.2 and PyTorch Lightning 1.9.5.
Use setuptools 77–80 for the legacy Lightning runtime; the graph extra includes
this constraint. The exact NN reference extra targets Python 3.10/3.11.
Install DGL and PyTorch Geometric extension wheels compatible with your PyTorch,
CUDA and operating-system versions before installing the graph extra. Consult
[PyG installation](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
and [DGL installation](https://www.dgl.ai/pages/start.html). A CPU-only setup is
supported. Avoid installing a different DGL major version without revalidation.

## Public GLACIER dependency

GLACIER uses the public MIST compatibility architecture and the following pinned
public helper implementation:

```bash
git clone https://github.com/coleygroup/ms-pred.git
cd ms-pred
git checkout 708148c2a8eb0c32120644436fefd2fe90cd2214
python -m pip install -e .
```

Follow that repository's dependency instructions. The supplied GLACIER weights
include layers absent from the standard upstream model at this revision;
`mist.glacier` provides their matching architecture with strict loading.
[The MIST compatibility notes](https://github.com/coleygroup/MIST/blob/main/docs/glacier.md)
explain the boundary.

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
msms-models doctor --config configs/molforge.yaml
```

`doctor` reports file presence and dependency availability. It does not certify
that a checkpoint matches an architecture: the model loader checks that during
inference. MCES additionally needs `[mces]` and an available CBC executable,
which the evaluator checks before computing any distances.
