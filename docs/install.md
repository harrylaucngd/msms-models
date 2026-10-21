# Installation

```bash
python -m pip install -e '.[mces]'
```

The core package covers everything that needs no neural network:
- data preparation and evaluation;
- MIST retrieval from a posterior cache;
- the nearest-neighbour and spectrum-blind baselines.

It needs Python ≥ 3.10, NumPy, SciPy, pandas, PyYAML and RDKit. `[mces]` adds
myopic-mces 1.0.1 and PuLP.

## Model environments

Some models pin mutually incompatible versions of NumPy, DGL, Lightning or
Transformers. Install each stack in its own environment, using the constraint file
of that stack, and run `msms-models` from the environment that matches the row.

| Stack | Models | Install |
|---|---|---|
| reference | `nearest-neighbor`; `mist` ranking for Table 3 | `pip install -e '.[nn,mces]'` (pins NumPy 1.24.4) |
| generation | `molforge`, `ms-gpt(-base)`, `frigid-base`, ScaledMIST encoding | `pip install -c environments/generation.txt -e '.[mist,molforge,frigid,ms-gpt,mces]'` |
| legacy graph | `iceberg`, `frigid`, `foam` | see below |
| glacier | `glacier` | `pip install -c environments/glacier.txt -e '.[mist,iceberg,glacier,mces]'` |
| specbridge | `dreams-chemberta` | `pip install -c environments/specbridge.txt -e '.[dreams-chemberta,mces]'` |
| glmr | `chemformer-two-stage` | `pip install -c environments/glmr.txt -e '.[chemformer-two-stage,mces]'` (GPU) |
| chemberta | `chemberta-classifier` | `pip install -e '.[chemberta,mces]'` |

**Reference stack.** The Table 3 MIST row breaks ties with NumPy 1.24.4's reverse
quicksort, as the original evaluation did, and later NumPy releases order ties
differently. The nearest-neighbour row uses the same NumPy, SciPy 1.10.1 and RDKit
2023.9.4 as the upstream implementation.

**Legacy graph stack** (ICEBERG, FRIGID refinement, FOAM): install PyTorch first, then
the torch-scatter wheel that matches it, then the extras:

```bash
python -m pip install 'torch==2.6.0'
python -m pip install torch-scatter -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
python -m pip install -c environments/legacy-graph.txt -e '.[frigid,iceberg,foam,mces]'
```

- DGL 1.1.3, torch-scatter 2.1.2 and PyTorch Lightning 1.9.5 are required, with setuptools below 81.
- If no prebuilt torch-scatter wheel matches your glibc, build it from source: `pip install --no-binary torch-scatter torch-scatter==2.1.2`.
- FRIGID needs NumPy 1 (BioNeMo), while GLACIER's ms-pred needs NumPy 2. Keep them in separate environments.
- MIST-conditioned rows read released posterior caches, so they need no MIST runtime.

## Checking an environment

```bash
msms-models doctor --config paper/table5/mist_molforge/config.yaml
```

`doctor` reports whether every asset is present, whether its pinned hash matches,
and whether each runtime import succeeds. It does not load weights; the model
loaders check checkpoint compatibility strictly at inference.

## Native CBC on macOS

PuLP's bundled CBC may be an Intel binary that cannot run on Apple Silicon. Install
a native one (`brew install cbc`) and pass `--mces-solver COIN_CMD` to
`msms-models evaluate`. The solver is recorded in the protocol and in the MCES cache
key.
