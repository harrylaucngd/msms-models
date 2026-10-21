# MS-GPT

`inference.py` implements the official SAFE decoder workflow; `adapter.py` validates and merges the learned LoRA adapter; `vendor/` contains decoder, density queries and ranking. `ms-gpt-base` rejects an adapter and `ms-gpt` requires one. Both use a matching encoder and training density calibration, with the default 20 queries × 5 samples.

Use the released decoder, tokenizer, benchmark-paired historical MIST encoder
and adapter together. The default configurations pin their encoder and decoder
hashes. The newer ScaledMIST checkpoints are different models.

## Download the released MSG assets

The [author-authorized Hugging Face mirrors](https://github.com/VIKI623/MS-GPT/issues/1)
provide the original files. From the repository root:

```bash
python -m pip install -e '.[ms-gpt]'
hf download nielsr/MS-GPT-MLM \
  --revision a139951c6cd2b6f31be9a53623a869610ee1d7ec \
  --local-dir checkpoints/ms-gpt/decoder
hf download nielsr/MS-GPT-MIST-MassSpecGym \
  --revision aecf3ae475a8dda28253136754b32619d454cac3 \
  --local-dir checkpoints/ms-gpt/msg-encoder
hf download nielsr/MS-GPT-SAFE-GPT-Base tokenizer.json \
  --revision 8d0560e93964a3344e724153c1bc83178d4a9d8f \
  --local-dir checkpoints/ms-gpt
hf download nielsr/MS-GPT-Posterior-Adapter-MassSpecGym \
  --revision bc02ce8df7cad8560ee0d5aeaa9e3f967569ca77 \
  --local-dir checkpoints/ms-gpt/adapters/msg
hf download nielsr/MS-GPT-MassSpecGym --repo-type dataset --include 'test/*' \
  --revision d3d016c452ffd61d6b0f2cc5948c1be2b3c904a0 \
  --local-dir data/ms-gpt/massspecgym
```

Prepare the ordinary MSG queries as described in the common data guide. The
annotated parquet files supply peak formulas; the encoder reads only spectrum
annotations and metadata, never target structures or target fingerprints.
The published training density band is included as `msg-calibration.json`.

For CANOPUS, use the corresponding repositories and destinations:

| Repository | Revision | Destination |
|---|---|---|
| `nielsr/MS-GPT-MIST-NPLIB1` | `9ee89f4adaa0a9e9e7d866e251475689b1149418` | `checkpoints/ms-gpt/canopus-encoder` |
| `nielsr/MS-GPT-Posterior-Adapter-NPLIB1` | `00028ffb8fefea9fc9d3a3a72abc63f907313922` | `checkpoints/ms-gpt/adapters/canopus` |
| `nielsr/MS-GPT-NPLIB1` (dataset, `test/*`) | `d345e71049bdcdd2df187272115377fcb2f6e41c` | `data/ms-gpt/nplib1` |

Reuse the shared decoder and tokenizer. The public NPLIB1 test export contains
672 spectra; it is not interchangeable with the project's archived 819-spectrum
CANOPUS evaluation. `canopus-calibration.json` contains the
published NPLIB1 band. Keep each benchmark's query IDs and known formulas aligned
to its annotated spectra.

## Assets

| Configuration key | Input |
|---|---|
| `decoder_checkpoint, decoder_config` | Official decoder weights and the included `configs/ms_gpt/architecture.yaml`. |
| `decoder_tokenizer` | Official SAFE tokenizer JSON. |
| `calibration` | Training-split density thresholds paired with the selected MIST encoder. |
| `adapter_checkpoint` | Trained PEFT directory for `ms-gpt`; omitted for `ms-gpt-base`. |

The default configurations use `mist_checkpoint` and `legacy_features` (the
released annotated parquet directory). [`encoders/mist/legacy.py`](../../../encoders/mist/legacy.py)
owns this encoder and its included architecture. Alternatively, supply `posterior`,
`posterior_ids` and `posterior_manifest`; the manifest must record the matching
`checkpoint_sha256`. The [data guide](../../../../../docs/data.md#mist-compact-features-and-posteriors)
defines the cache format. Changing the encoder requires recalibration and, for
the adapted model, retraining the adapter.

Provider links and current download availability are in the [asset inventory](../../../../../docs/assets.md). Paths are explicit in the configurations below.

## Run

From the repository root, after the [common data setup](../../../../../docs/data.md) and [asset preparation](../../../../../docs/assets.md):

```bash
python -m pip install -e '.[ms-gpt]'
msms-models doctor --config configs/ms_gpt/msg-base.yaml
msms-models predict --config configs/ms_gpt/msg-base.yaml --limit 2 --output results/smoke/ms-gpt-base
msms-models predict --config configs/ms_gpt/msg-base.yaml
msms-models evaluate --task de-novo --queries data/msg/queries.jsonl \
  --predictions results/ms-gpt-base/msg/predictions.jsonl \
  --output results/ms-gpt-base/msg/metrics.json --mces
```

`--limit 2` exercises the configured inference on the first two queries; it is a plumbing check, not a benchmark result. Choose a fresh output directory for each run.

## Variants and source

| Model ID | Configurations | Entry point |
|---|---|---|
| `ms-gpt-base` | [msg](../../../../../configs/ms_gpt/msg-base.yaml), [canopus](../../../../../configs/ms_gpt/canopus-base.yaml) | [inference.py](inference.py) |
| `ms-gpt` | [msg](../../../../../configs/ms_gpt/msg.yaml), [canopus](../../../../../configs/ms_gpt/canopus.yaml) | [inference.py](inference.py) |

Assets, parameters and runtime imports are registered in [models/\_\_init\_\_.py](../../__init__.py). Shared loading, fingerprint ranking and evaluation remain in the library core instead of being copied per method. [Third-party notices](../../../../../THIRD_PARTY_NOTICES.md) record the source revisions and licenses.
