# MSNovelist: Morgan-4096/MIST adaptation

This package integrates the locally adapted **ctMSNovelist** decoder, not the
original CSI:FingerID model. Training uses true radius-2, 4,096-bit Morgan
fingerprints and molecular formulas; prediction uses MIST posterior probabilities
and supplied formulas. It does not infer formulas from spectra.

The architecture, 39-token vocabulary, formula order
`C F I Cl N O P Br S H`, fingerprint `torch.round` behavior (including 0.5 ties),
and parameter names are retained. Original 3,609-bit CSI:FingerID checkpoints,
MS2SMILES and co-training/TransVAE checkpoints are not supported.

## Installation

```bash
python -m pip install -e '.[msnovelist]'
```

The optional extra adds PyTorch >=2.6; RDKit and NumPy are core dependencies.
Use the PyTorch distribution appropriate for your CPU/CUDA system. No Java,
SIRIUS wrapper, JPype, torchvision or torchaudio is needed. Imports and `--help`
do not load weights, open databases, or start training.

## External assets

These assets are **not included in Git or the existing public asset bundle**.
Obtain the adapted checkpoint and prepared database from the contributor; upstream
ctMSNovelist's published CSI:FingerID assets are not replacements. Until those
adapted assets are distributed separately, the examples require contributor access.

| Asset | Required for | Description |
|---|---|---|
| `decoder_checkpoint` | prediction | Raw adapted model state dictionary, loaded strictly with `weights_only=True`. Reference: `w-12-0.034-0.019.pth` (~17 MB). |
| `posterior`, `posterior_ids` | cached prediction | NumPy `(N, 4096)` finite probabilities in `[0,1]`, and matching string spectrum IDs. Rows are aligned to query IDs by the shared MIST loader. |
| Prepared SQLite database | training | `org_moses_0824.db` (~13 GB), containing stored fingerprints, structures, formulas and folds. |

Reference checkpoint SHA-256:
`2c214d6be08953a7616025b0dd7c47bb91fbb66bbe48ad9c2f1c4a794a86e425`.

Prepared minimal database SHA-256:
`4e1b900505d3745e2f59169b7ddbf44825156885b152b8218b8d7ba63c8ee87f`.

Contributor source location on `molgpu02`:
`~/foam_baselines/baselines/ctMSNovelist/`. The checkpoint is under
`weights_folder/no_simFP_test_full_shuffleTrue_fullfp/0/`; the prepared database
used for this integration is `ctMSNovelist_minimal/MS_DATA/org_moses_0824.db`.

Instead of caches, configure `mist_checkpoint` and `compact_cache` and install
`.[mist,msnovelist]` to use the shared ScaledMIST encoder. See the
[MIST guide](../../../encoders/mist/README.md). Different encoders/posteriors can
change results; these examples do not establish parity with the 2025 notebooks.

## Prediction

Edit the paths in [CANOPUS](../../../../../configs/msnovelist/canopus.yaml) or
[MSG](../../../../../configs/msnovelist/msg.yaml), then run from the repository root:

```bash
msms-models predict --config configs/msnovelist/canopus.yaml --device cuda:0
msms-models evaluate --protocol table5 \
  --queries results/msnovelist/canopus/queries.jsonl \
  --predictions results/msnovelist/canopus/predictions.jsonl \
  --output results/msnovelist/canopus/metrics.json
```

`predict(config, queries)` returns one record per query with `id`, `candidates`,
`scores` and `status`. Query formulas must be neutral formulas containing only the
supported elements. Missing IDs, incompatible fingerprints/formulas or checkpoint
shapes raise errors rather than silently substituting inputs.

| Parameter | Default | Meaning |
|---|---|---|
| `beam_size` | 100 | Native beam width and number of traced candidates. |
| `top_k` | 10 | Maximum number of surviving unique candidates (`<= beam_size`). |
| `batch_size` | 32 | Queries per decoder batch; the final partial batch is supported. |

The decoder keeps the native 128 decoding steps, temperature 1, and disabled
negative-count pruning. Beam search bookkeeping remains on CPU, while neural
model operations use the requested device. This preserves the source algorithm;
it is not a throughput optimization.

Candidates are parsed/canonicalized with the repository's RDKit helpers, with
stereochemistry removed. Invalid/empty molecules and non-finite beam scores are
excluded. Deduplication uses connectivity InChIKey-14, retaining the highest score;
score ties retain beam order. Scores are native accumulated log-probabilities
(higher is better). Return up to `top_k` candidates; an empty result still keeps
its query record. **There is no final formula filter or fingerprint reranking.**
The notebooks' formula checks only gated auxiliary fingerprint calculations.

RDKit replaces the notebooks' Java/SIRIUS normalization. This is an intentional
behavior change: validity, identity and ranking may differ from historical results.
The shared evaluator is used instead of notebook-specific metrics and indexing.

## Training

Training starts from fresh weights, using the existing prepared database:

```bash
python -m msms_models.models.de_novo.msnovelist.train \
  --database /path/to/org_moses_0824.db \
  --output runs/msnovelist-fold0 \
  --fold 0 --epochs 20 --batch-size 256 --device cuda:0 --workers 4 --seed 42
```

Defaults are fold 0, 20 epochs, batch size 256, CPU, zero data-loader workers and
seed 42. Fold 0 is validation; the other stored folds are training (analogously
for another `--fold`). Training shuffles the selected rows and uses Adam's default
learning rate of 0.001, SMILES cross entropy plus `0.03 * hydrogen MSE`, and **no
fingerprint-error simulation**. Validation reports SMILES cross entropy alone,
matching the original helper. Loss reporting averages batch losses as in the source.

The database is opened read-only. Its `compounds` table must contain:

| Column | Encoding |
|---|---|
| `id` | Unique integer identifier. |
| `fp` | Exactly 4,096 **unpacked** uint8 binary values; not 512 packed bytes. |
| `smiles` | Valid SMILES representable by the fixed vocabulary, at most 126 tokens before start/end markers. |
| `mf` | Legacy pickled `collections.Counter`/dict of nonnegative integer element counts. The loader permits only the Counter pickle global. |
| `grp` | Stored `fold0` through `fold9` assignments. |
| `perm_order` | Ordering key; ties are resolved by ID. |

Use the supplied prepared database to retain its split. The exploratory data
notebook assigned folds randomly without a recorded seed; a new database cannot
be claimed to recreate the original split. This package does not rebuild that
13 GB database or certify its historical train/test overlap.

The output directory must not already exist. It receives the resolved `config.json`,
raw `w-EPOCH-TRAINLOSS-VALLOSS.pth` states usable by prediction, and
`optimizer-EPOCH.pth` states. Automatic resume is not provided. Choose a batch size
that leaves no singleton training batch, since the encoder uses BatchNorm.
Unsupported structures and non-finite losses fail explicitly. Seeds are recorded,
but exact numerical reproducibility across devices/library versions is not promised.

## Provenance and validation

The local source is based on `mrunalimanj/foam_baselines` revision
`2aed7c9c67c5ccf32adc032bc099b2a403106fd5`, under `baselines/ctMSNovelist`, with local
configuration/training edits and a separate `ctMSNovelist_minimal` training entry.
The model files in the minimal and parent copies matched when inspected.
ctMSNovelist is attributed to Kai Zhao, Yanmin Liu, Longyang Dian, Shiwei Sun and
Xuefeng Cui, *How to Train Your Neural Network for Molecular Structure Generation
from Mass Spectra?* See the retained [MIT license](vendor/LICENSE.txt) and
[third-party notices](../../../../../THIRD_PARTY_NOTICES.md).

Packaging changes include local imports, explicit config, nonpersistent device-aware
buffers, device-correct temporary tensors and a read-only database loader. Model
math, including the original detached encoder starting states in training, is retained.
Unused co-training, alternate beam search, Java utilities and notebook metrics are omitted.

```bash
python -m pip install -e '.[msnovelist,test]'
MSNOVELIST_REFERENCE_CHECKPOINT=/path/to/w-12-0.034-0.019.pth \
  python -m pytest tests/test_msnovelist.py tests/test_registry.py
```

Tests cover data/token contracts, ranking and mocked prediction orchestration,
toy-model loss handling, model construction/device buffers, and strict checkpoint
loading when the external checkpoint is supplied. They do not run the molecular
model's training or prediction. GPU execution and benchmark reproduction are not
validated by these tests.
