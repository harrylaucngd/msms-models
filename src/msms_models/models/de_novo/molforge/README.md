# MolForge

`inference.py` contains the MIST adapter, strict checkpoint loading and beam invocation. `vendor/transformer.py`, `vendor/decoder.py` and `vendor/parameters.py` contain the complete local decoder inference implementation. The source and target SentencePiece models are both required. Scores in exported predictions are negative native beam costs, so higher is better; beam order is unchanged.

The decoder consumes thresholded 4,096-bit MIST fingerprints. Calibrate the threshold for the selected encoder. The default output preserves raw beam slots, including invalid strings; formula filtering is an explicit option.

The included decoder is CC BY-NC-SA 4.0; see `vendor/LICENSE.txt`.

## Assets

| Configuration key | Input |
|---|---|
| `decoder_checkpoint` | Original MolForge transformer weights from the upstream project. |
| `source_tokenizer` | Fingerprint-index SentencePiece model (`combined_morgan4096_vocab_sp.model`). |
| `target_tokenizer` | SMILES SentencePiece model (`combined_smiles_vocab_sp_morgan4096.model`). |

Fingerprint conditioning uses `mist_checkpoint` and `compact_cache`. Alternatively, supply `posterior` and `posterior_ids` to reuse a cache without loading MIST. The [data guide](../../../../../docs/data.md#mist-posteriors) defines these formats.

**Parameters.**

| Key | Default | Meaning |
|---|---|---|
| `fingerprint_threshold` | none | posterior threshold for the on-bits; calibrate it for the encoder |
| `beam_size` | 10 | beams decoded per spectrum |
| `source_length`, `target_length` | 104, 130 | maximum fingerprint-token and SMILES-token lengths |
| `formula_filter` | `false` | `true` drops beams whose formula differs from the query |

The encoder keys (`encoder_batch_size`, `attention_padding`) are in the [MIST guide](../../../encoders/mist/README.md).

Provider links and current download availability are in the [asset inventory](../../../../../docs/assets.md). Paths are explicit in the configurations below.

## Run

From the repository root, after the [common data setup](../../../../../docs/data.md) and [asset preparation](../../../../../docs/assets.md):

```bash
bash environments/install.sh   # from the repository root
msms-models predict --config paper/table5/mist_molforge/config.yaml --limit 2 --output results/smoke/molforge
msms-models predict --config paper/table5/mist_molforge/config.yaml
msms-models evaluate --protocol table5 --queries results/paper/table5/mist_molforge/queries.jsonl \
  --predictions results/paper/table5/mist_molforge/predictions.jsonl \
  --output results/paper/table5/mist_molforge/metrics.json
```

`--limit 2` exercises the configured inference on the first two queries; it is a plumbing check, not a benchmark result.

## Variants and source

| Model ID | Configurations | Entry point |
|---|---|---|
| `molforge` | [msg](../../../../../paper/table5/mist_molforge/config.yaml), [canopus](../../../../../configs/molforge/canopus.yaml) | [inference.py](inference.py) |

Shared loading, fingerprint ranking and evaluation remain in the library core instead of being copied per method. [Third-party notices](../../../../../THIRD_PARTY_NOTICES.md) record the source revisions and licenses.
