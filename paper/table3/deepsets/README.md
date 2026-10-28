# Table 3 · DeepSets, DeepSets + Fourier features

| | Hit@1 | Hit@5 | Hit@20 | MCES@1 |
|---|---|---|---|---|
| DeepSets [8] | 4.42% (3.92–4.97) | 14.46% (13.58–15.36) | 30.76% (29.67–31.93) | 15.04 (14.89–15.19) |
| DeepSets + Fourier features [8] | 6.56% (5.95–7.16) | 16.46% (15.58–17.35) | 33.46% (32.39–34.59) | 14.14 (13.98–14.31) |

Intervals are 99.9% BCa over 20,000 spectrum-level resamples.

**Source.** [`models/retrieval/deepsets.py`](../../../src/msms_models/models/retrieval/deepsets.py) and [`massspecgym/`](../../../src/msms_models/massspecgym/) ([MassSpecGym](https://github.com/pluskal-lab/MassSpecGym)).

**What runs.** Each spectrum becomes 61 peak rows, the precursor first, zero-padded (`SpecTokenizer`: peaks from m/z 10, the 60 most intense, relative intensities). Every row passes through φ, the rows are summed, and ρ predicts a 4,096-bit Morgan fingerprint; candidates are ranked by cosine similarity. With Fourier features, m/z is expanded by DreaMS's frequencies before φ. `run.sh` trains the model first: cosine loss, Adam at 10⁻³, a global batch of 1,024, two validations per epoch, early stopping after three without improvement, and the checkpoint with the lowest validation loss. The plain row reads peaks up to m/z 1000, the Fourier row up to 1005.

**Inputs.** Official MassSpecGym training, validation and test queries and the original formula candidate pool (`bash paper/prepare_data.sh`).

## Run

From the repository root:

```bash
bash paper/table3/deepsets/run.sh
```

Set `DEVICE=cuda:0` for GPU training and inference and `OUT=...` to choose the output directory.
`run.sh` predicts and evaluates with the fixed `table3` protocol.
