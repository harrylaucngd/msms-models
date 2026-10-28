# MolForge

MolForge is a transformer that translates a fingerprint into SMILES. Here it decodes
thresholded 4,096-bit MIST fingerprints: the on-bits are tokenized with a fingerprint-index
SentencePiece model, and beam search writes SMILES tokens. [`inference.py`](inference.py)
thresholds the MIST posterior, loads the checkpoint strictly and runs the beam search;
`vendor/transformer.py`, `vendor/decoder.py` and `vendor/parameters.py` contain the decoder.
Predictions keep every beam, invalid strings included, and score each by its negative beam
cost, so higher is better. Formula filtering is an option.

## Assets

| Configuration key | Input |
|---|---|
| `decoder_checkpoint` | MolForge transformer weights. |
| `source_tokenizer` | Fingerprint-index SentencePiece model (`combined_morgan4096_vocab_sp.model`). |
| `target_tokenizer` | SMILES SentencePiece model (`combined_smiles_vocab_sp_morgan4096.model`). |
| `posterior`, `posterior_ids` | MIST posteriors, or `mist_checkpoint` + `diffms_features` to encode with MistNet; see the [MIST guide](../../../encoders/mist/README.md). |

**Parameters.**

| Key | Default | Meaning |
|---|---|---|
| `fingerprint_threshold` | none | posterior threshold for the on-bits, calibrated for the encoder |
| `beam_size` | 10 | beams decoded per spectrum |
| `source_length`, `target_length` | 104, 130 | maximum fingerprint-token and SMILES-token lengths |
| `formula_filter` | `false` | `true` drops beams whose formula differs from the query |

The encoder keys (`encoder_batch_size`, `attention_padding`) are in the [MIST guide](../../../encoders/mist/README.md).

## Run

From the repository root, after `bash paper/prepare_data.sh`:

```bash
bash paper/table5/mist_molforge/run.sh           # MIST 4M-aug posteriors
bash paper/table5/mist_vdiffms_molforge/run.sh   # MIST vDiffMS, batch size 24 and 1
```

The configurations are in [`paper/table5/mist_molforge/`](../../../../../paper/table5/mist_molforge/)
and [`paper/table5/mist_vdiffms_molforge/`](../../../../../paper/table5/mist_vdiffms_molforge/).
Checkpoints are listed in the [asset inventory](../../../../../docs/assets.md).

## Source

[knu-lcbc/MolForge](https://github.com/knu-lcbc/MolForge), **CC BY-NC-SA 4.0**; see
[`vendor/LICENSE.txt`](vendor/LICENSE.txt) and the
[third-party notices](../../../../../THIRD_PARTY_NOTICES.md).
