# SMILES and SELFIES Transformer

MassSpecGym's de novo baselines (Table 5: SMILES Transformer, SELFIES Transformer), with
the formula of the bonus challenge. An `nn.Transformer` encodes the tokenized spectrum (the
precursor, then up to 60 peaks as (m/z, intensity), m/z 10–1005) and decodes a molecule as
byte-level BPE SMILES tokens or SELFIES tokens. The two rows are one model with two
tokenizers.

| File | Contents |
|---|---|
| [`model.py`](model.py) | the network and its decoding loop |
| [`tokenizers.py`](tokenizers.py) | the BPE and SELFIES tokenizers, saved as `tokenizer.json` |
| [`train.py`](train.py) | builds the tokenizer and trains with MassSpecGym's settings |
| [`inference.py`](inference.py) | `predict(config, queries)` |

**Behaviour kept from upstream.**
- **Network:** there is no positional encoding. Peak and token embeddings are scaled by √d_model. The 118 element counts of the formula pass through one linear layer and are added to every peak before that scaling. Padded peaks are masked in the encoder but not in the decoder's cross-attention.
- **Training:** the loss is cross entropy over every target position, padding included.
- **Prediction:** each spectrum's 10 candidates are 10 independent samples at temperature 1, in sampling order; a single prediction is decoded greedily. A batch stops decoding only when every sequence emits `</s>` at the same step, and tokens generated after a sequence's `</s>` stay in its string unless they are special tokens.
- **Tokenizers:** the BPE tokenizer is trained with the library defaults on the 4M MassSpecGym molecules, then on the training molecules (one per training spectrum). The SELFIES vocabulary is the sorted alphabet of the training molecules. Sequences are padded or truncated to 200 tokens (SMILES) and 150 (SELFIES).

Against MassSpecGym's own code with the same weights, the tokenizers, logits and greedy
decoding are identical, and so are the samples under the same PyTorch; PyTorch 2.7, locked
here, draws different `torch.multinomial` samples from the same seed than MassSpecGym's 2.3.
MassSpecGym released no checkpoints, so a model has to be trained first. Retrained models
match the published rows within their confidence intervals, not to the digit.

## Assets

| Configuration key | Input |
|---|---|
| `checkpoint` | `model.pt` from `train.py`; `tokenizer.json` is read from the same folder |

**Parameters.** `k_predictions` (10) samples per spectrum, `temperature` (1.0),
`batch_size` (64) spectra decoded together, as in MassSpecGym's test runs.

## Run

```bash
bash environments/install.sh   # from the repository root
bash paper/table5/smiles_transformer/run.sh
```

## Training

```bash
python -m msms_models.models.de_novo.smiles_transformer.train --tokenizer smiles-bpe \
  --train data/msg/train.jsonl --validation data/msg/val.jsonl \
  --molecules data/MassSpecGym_molecules_MCES2_disjoint_with_test_fold_4M.tsv \
  --output checkpoints/smiles_transformer/smiles --device cuda:0
```

`--tokenizer selfies` needs no `--molecules`. The defaults are MassSpecGym's "v3"
formula-challenge runs (`scripts/run.sh`): d_model 256, 4 decoder layers, dropout 0.1, Adam
at 3e-4, a global batch of 512 (8 GPUs × 64), one validation per epoch, and early stopping
after 3 epochs without a lower validation loss. SMILES uses 4 heads and 3 encoder layers;
SELFIES uses 8 heads and 6 encoder layers. Every setting can be overridden with a flag, e.g.
`--batch-size`.

## Source

[pluskal-lab/MassSpecGym](https://github.com/pluskal-lab/MassSpecGym) at `485deb22df`
(`models/de_novo/smiles_tranformer.py`, `models/tokenizers.py`, `scripts/run.py`), MIT
license; see [`massspecgym/vendor/LICENSE.txt`](../../../massspecgym/vendor/LICENSE.txt).
Lightning and PyTorch Geometric are replaced by
[`massspecgym/training.py`](../../../massspecgym/training.py) and a plain linear layer.
