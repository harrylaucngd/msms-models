# MSNovelist (Morgan-4096, MIST-conditioned)

A ctMSNovelist decoder trained on radius-2, 4,096-bit Morgan fingerprints and formulas,
conditioned at prediction time on MIST posteriors, which it rounds at 0.5. Beam search
(100 beams, 128 steps) writes SMILES; candidates are canonicalized with RDKit
(stereochemistry removed), invalid ones are dropped, and the best-scoring candidate per
InChIKey-14 is kept, up to `top_k`. Scores are accumulated log-probabilities. There is no
formula filter. A formula with an element outside the decoder's vocabulary
(`C F I Cl N O P Br S H`) gets an empty prediction with status `unsupported_formula`.

| File | Contents |
|---|---|
| [`inference.py`](inference.py) | MIST posteriors, formula vectors, beam search and candidate ranking |
| [`build_database.py`](build_database.py) | writes the SQLite training database from a SMILES list |
| [`data.py`](data.py) | the fold datasets read by training |
| [`train.py`](train.py) | trains the decoder from fresh weights |
| [`vendor/`](vendor/) | the ctMSNovelist model, vocabulary and beam search |

## Assets

| Configuration key | Input |
|---|---|
| `decoder_checkpoint` | the trained decoder, `checkpoints/msnovelist/decoder.pth` |
| `mist_checkpoint`, `diffms_features` | the MIST released with DiffMS (`checkpoints/mist_vdiffms/mist_msg.pt`) and DiffMS subformula trees; or `posterior`, `posterior_ids` for a posterior cache. See the [MIST guide](../../../encoders/mist/README.md). |

**Parameters.** `beam_size` (100), `top_k` (10) candidates kept, `batch_size` (32)
spectra per decoder batch. The paper row encodes with `attention_padding: additive-bool`
(the boolean padding mask added to the attention scores, paper §6.1.1) and
`encoder_batch_size: 128`.

## Training

The decoder is trained from public molecules. `build_database.py` turns a SMILES list into
a SQLite `compounds` table: each molecule's non-isomeric RDKit SMILES, its unpacked Morgan
radius-2/4096 fingerprint, its formula counts and one of ten random folds. Molecules the
decoder cannot write (tokens outside its vocabulary, elements outside its formula
vocabulary, more than 126 tokens) are skipped, and duplicates are removed.

```bash
python -m msms_models.models.de_novo.msnovelist.build_database \
  --smiles fp2mol_train.txt --output data/msnovelist/train.db
python -m msms_models.models.de_novo.msnovelist.train --database data/msnovelist/train.db \
  --output checkpoints/msnovelist/run --fold 0 --epochs 20 --batch-size 256 --device cuda:0
cp checkpoints/msnovelist/run/w-<epoch with the lowest validation loss>.pth checkpoints/msnovelist/decoder.pth
```

The selected fold is validation and the others are training. The loss is SMILES cross
entropy plus 0.03 × hydrogen-count MSE, with Adam at its default learning rate. Each epoch
writes `w-EPOCH-TRAIN-VAL.pth` and the optimizer state. The paper row trains on the
decoder pretraining set of [DiffMS](https://github.com/coleygroup/DiffMS); see the
[row README](../../../../../paper/table5/mist_msnovelist/README.md).

## Run

From the repository root, after `bash paper/prepare_data.sh` and training the decoder:

```bash
bash paper/table5/mist_msnovelist/run.sh
```

The configuration is [`paper/table5/mist_msnovelist/config.yaml`](../../../../../paper/table5/mist_msnovelist/config.yaml).

## Source

`mrunalimanj/foam_baselines` at `2aed7c9`,
`baselines/ctMSNovelist`, adapted from ctMSNovelist (Zhao, Liu, Dian, Sun and Cui), MIT;
see [`vendor/LICENSE.txt`](vendor/LICENSE.txt) and the
[third-party notices](../../../../../THIRD_PARTY_NOTICES.md).
