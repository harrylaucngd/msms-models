# Table 3 · Fingerprint FFN

| | Hit@1 | Hit@5 | Hit@20 | MCES@1 |
|---|---|---|---|---|
| Fingerprint FFN [8] | 5.09% (4.57–5.66) | 14.69% (13.83–15.56) | 31.97% (30.86–33.10) | 14.94 (14.79–15.09) |

Intervals are 99.9% BCa over 20,000 spectrum-level resamples.

**Source.** [`models/retrieval/fingerprint_ffn.py`](../../../src/msms_models/models/retrieval/fingerprint_ffn.py) and [`massspecgym/`](../../../src/msms_models/massspecgym/), reimplementing [MassSpecGym](https://github.com/pluskal-lab/MassSpecGym)'s `FingerprintFFNRetrieval` at `e5080d6126`. The printed row is MassSpecGym's.

**What runs.** Each spectrum is binned at 1 Da up to m/z 1005 (`SpecBinner`: peaks from m/z 10, relative intensities summed per bin). A seven-layer MLP (128 hidden units, dropout 0.1) predicts a 4,096-bit Morgan fingerprint through a sigmoid; candidates are ranked by cosine similarity. MassSpecGym released no checkpoints, so `run.sh` trains the model with the settings of MassSpecGym's own run; a retrained model reproduces the printed numbers within their confidence intervals, not to the digit. Training: cosine loss, Adam at 10⁻³, a global batch of 512 (MassSpecGym's 8 GPUs × 64), two validations per epoch, early stopping after three without improvement, and the checkpoint with the lowest validation loss.

**Inputs.** Official MassSpecGym training, validation and test queries and the original formula candidate pool (`bash paper/prepare_data.sh`).

**Environment.** The locked environment (`bash environments/install.sh`; see [installation](../../../docs/install.md)). A GPU is recommended for training.

## Run

From the repository root:

```bash
bash paper/table3/fingerprint_ffn/run.sh
```

Set `DEVICE=cuda:0` for GPU training and inference and `OUT=...` to choose the output directory.
`run.sh` predicts and evaluates with the fixed `table3` protocol.
