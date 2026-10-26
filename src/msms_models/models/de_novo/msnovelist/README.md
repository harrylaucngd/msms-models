# MSNovelist (Morgan-4096, MIST-conditioned)

The MIST + MSNovelist baseline cited in Table 5 (from the DiffMS paper). It is a
ctMSNovelist decoder retrained on radius-2, 4,096-bit Morgan fingerprints and formulas,
and conditioned at prediction time on MIST posteriors, which it rounds at 0.5. It is
not the original CSI:FingerID MSNovelist, whose checkpoints are incompatible.

**Encoder.** The original baseline used the DiffMS-era MIST (Bohde's
`msg_magma_best.ckpt` for MassSpecGym, `canopus_fp_mist_4096` for CANOPUS), run at batch
size 128. The configurations read exactly those posteriors. Other encoders' posteriors can
be supplied, but they define a different model.

**What runs.** Beam search with 100 beams and 128 steps. Candidates are canonicalized with
RDKit (stereochemistry removed), invalid ones are dropped, and the best-scoring candidate
per InChIKey-14 is kept, up to `top_k`. Scores are accumulated log-probabilities. There is
no formula filter. The original notebooks normalized with Java/SIRIUS instead of RDKit,
which can change which candidates count as valid or identical. A formula with an element
outside the decoder's vocabulary (`C F I Cl N O P Br S H`; two MassSpecGym test spectra
contain Si) gets an empty prediction with status `unsupported_formula`.

## Assets

| Configuration key | Input |
|---|---|
| `decoder_checkpoint` | the adapted checkpoint `w-12-0.034-0.019.pth` (17 MB) |
| `posterior`, `posterior_ids` | the encoder's posterior cache and its spectrum IDs |

Both come from the contributor and are not yet in the Hugging Face release; see the
[asset inventory](../../../../../docs/assets.md).

**Parameters.** `beam_size` (100), `top_k` (10) candidates kept, `batch_size` (32)
spectra per decoder batch.

## Run

```bash
bash environments/install.sh   # from the repository root
msms-models predict --config configs/msnovelist/msg.yaml --device cuda:0
msms-models evaluate --task de-novo --queries results/msnovelist/msg/queries.jsonl \
  --predictions results/msnovelist/msg/predictions.jsonl --output results/msnovelist/msg/metrics.json
```

## Training

Training starts from fresh weights on the contributor's prepared SQLite database
(`org_moses_0824.db`, 13 GB), whose `compounds` table holds unpacked 4,096-bit
fingerprints, SMILES, pickled formula counters and the stored `fold0`–`fold9` assignment:

```bash
python -m msms_models.models.de_novo.msnovelist.train --database org_moses_0824.db \
  --output runs/msnovelist-fold0 --fold 0 --epochs 20 --batch-size 256 --device cuda:0
```

The selected fold is validation and the others are training. The loss is SMILES cross
entropy plus 0.03 × hydrogen-count MSE, with Adam at its default learning rate and no
fingerprint-error simulation. Each epoch writes `w-EPOCH-TRAIN-VAL.pth`, usable for
prediction, and the optimizer state.

## Source

`mrunalimanj/foam_baselines` at `2aed7c9`, `baselines/ctMSNovelist`, adapted from
ctMSNovelist (Zhao, Liu, Dian, Sun and Cui, MIT license; see
[`vendor/LICENSE.txt`](vendor/LICENSE.txt) and the
[third-party notices](../../../../../THIRD_PARTY_NOTICES.md)). The model, vocabulary and
beam search are unchanged; imports and devices were made package-local.
