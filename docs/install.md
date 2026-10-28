# Installation

Every model and the evaluator run in one locked environment: Linux x86-64, Python 3.11
and PyTorch 2.7.0 with CUDA 12.8. All package versions are fixed in
[`environments/linux-cuda.txt`](../environments/linux-cuda.txt), and
[`environments/install.sh`](../environments/install.sh) builds it with
[uv](https://docs.astral.sh/uv/):

```bash
bash environments/install.sh          # creates .venv
source .venv/bin/activate
```

Versions that matter for results:

| Package | Version | Why |
|---|---|---|
| NumPy | 2.2.6 | FRIGID's length model is a NumPy 2 pickle (the lock overrides bionemo-moco's `numpy<2`) |
| RDKit | 2025.3.6 | canonical SMILES, InChIKeys and fingerprints; ms-pred (GLACIER) requires 2025.03 |
| Transformers | 4.49.0 | the FRIGID and MS-GPT decoders |
| PuLP | 3.3.2 | PuLP 4 removed an API that myopic-mces 1.0.1 calls |
| DGL | 1.1.3 (CUDA 12.1 build) | ICEBERG and GLACIER graphs |

To change a version, edit the pins in [`environments/lock.sh`](../environments/lock.sh)
and run it.

## Evaluation only

Preparing data and evaluating predictions needs no neural network, and works on any
platform:

```bash
python -m pip install -e '.[mces]'
```

This installs current releases rather than the locked versions; CI tests both.

**Native CBC on macOS.** PuLP's bundled CBC may be an Intel binary that cannot run on
Apple Silicon. Install a native one (`brew install cbc`) and pass
`--mces-solver COIN_CMD` to `msms-models evaluate`.
