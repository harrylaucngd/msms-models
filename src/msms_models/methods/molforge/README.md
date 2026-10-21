# MolForge

`inference.py` contains the MIST adapter, strict checkpoint loading and beam invocation. `vendor/transformer.py`, `vendor/decoder.py` and `vendor/parameters.py` contain the complete local decoder inference implementation. The source and target SentencePiece models are both required. Scores in exported predictions are negative native beam costs, so higher is better; beam order is unchanged.

The decoder consumes thresholded 4,096-bit MIST fingerprints. Calibrate the threshold for the selected encoder. The default output preserves raw beam slots, including invalid strings; formula filtering is an explicit option. Checkpoint and tokenizer identities determine which decoder is being evaluated.

The included decoder is CC BY-NC-SA 4.0; see `vendor/LICENSE.txt`.

## Assets

| Configuration key | Input |
|---|---|
| `decoder_checkpoint` | Original MolForge transformer weights from the upstream project. |
| `source_tokenizer` | Fingerprint-index SentencePiece model (`combined_morgan4096_vocab_sp.model`). |
| `target_tokenizer` | SMILES SentencePiece model (`combined_smiles_vocab_sp_morgan4096.model`). |

Fingerprint conditioning uses `mist_checkpoint` and `compact_cache`. Alternatively, supply `posterior`, `posterior_ids` and `posterior_manifest` to reuse a validated cache without loading MIST. The [data guide](../../../../docs/data.md#mist-compact-features-and-posteriors) defines these formats.

Provider links and current download availability are in the [asset inventory](../../../../docs/assets.md). Paths are explicit in the configurations below.

## Run

From the repository root, after the [common data setup](../../../../docs/data.md) and [asset preparation](../../../../docs/assets.md):

```bash
python -m pip install -e '.[mist,molforge]'
msms-models doctor --config configs/molforge/msg.yaml
msms-models smoke --config configs/molforge/msg.yaml --output results/smoke/molforge --limit 2
msms-models infer --config configs/molforge/msg.yaml
msms-models evaluate --task de-novo --queries data/msg/queries.jsonl \
  --predictions results/molforge/msg/predictions.jsonl \
  --output results/molforge/msg/metrics.json --mces
```

The smoke command uses the configured inference budget on a small subset. For a cheaper developmental check, copy the configuration and explicitly reduce the budget; retain that configuration with the report. Choose a fresh output directory on each run.

## Variants and source

| Model ID | Configurations | Entry point |
|---|---|---|
| `molforge` | [msg](../../../../configs/molforge/msg.yaml), [canopus](../../../../configs/molforge/canopus.yaml) | [inference.py](inference.py) |

[spec.py](spec.py) declares required assets, optional imports and accepted parameters. Shared loading, fingerprint ranking and evaluation remain in the library core instead of being copied per method. [Third-party notices](../../../../THIRD_PARTY_NOTICES.md) record the source revisions and licenses.
