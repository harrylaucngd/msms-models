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

The environment has been built from these files and tested on RHEL 8 (glibc 2.28) and
Ubuntu 20.04 (glibc 2.31), and every paper row has been run in it on an RTX A5000.

Versions that change results are fixed deliberately:

| Package | Version | Why |
|---|---|---|
| NumPy | 2.2.6 | FRIGID's length model can only be unpickled by NumPy 2. bionemo-moco declares `numpy<2`, so the lock overrides it, as FRIGID's own runs did. |
| RDKit | 2025.3.6 | canonical SMILES, InChIKeys and fingerprints; ms-pred (GLACIER) requires 2025.03 |
| Transformers | 4.49.0 | the FRIGID and MS-GPT decoders |
| PuLP | 3.3.2 | PuLP 4 removed an API that myopic-mces 1.0.1 calls |
| DGL | 1.1.3 (CUDA 12.1 build) | ICEBERG and GLACIER graphs |

Two results do not depend on the environment by construction:
- **MIST's Table 3 tie order** is NumPy 1.24's argsort. From NumPy 1.25, argsort breaks
  ties differently on AVX-512 CPUs, so `fingerprint.numpy124_argsort` reproduces the
  1.24 order exactly on any version.
- **torch-scatter** is not installed: its wheels for PyTorch 2.7 need glibc 2.32.
  `simulators/scatter.py` gives ICEBERG and GLACIER the same reductions from PyTorch,
  bit for bit.

To change a version, edit the pins in [`environments/lock.sh`](../environments/lock.sh),
run it, and check the affected rows again.

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
