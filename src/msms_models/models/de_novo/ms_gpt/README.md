# MS-GPT

MS-GPT's SAFE molecule language model is queried with ScaledMIST fingerprint
posteriors. For each spectrum the posterior is thresholded at 20 points of a
density-calibrated band to give 20 binary fingerprint queries. Five molecules are
sampled per query, filtered by the known formula, grouped by InChIKey-14 and ranked
by frequency, then by Tanimoto to the queries.

- [`inference.py`](inference.py) loads the decoder strictly and runs the querying and ranking.
- `vendor/` holds MS-GPT's decoder (`model.py`), density queries (`queries.py`) and ranking (`ranking.py`).

**Encoder.** MS-GPT is conditioned only on **ScaledMIST** (here MIST *4M-aug*). The MIST
architecture released with MS-GPT and FRIGID is not supported. Its posteriors carry the
problems described in *MassSpecGym in the Wild* (§6.1). MS-GPT's posterior-aligned LoRA
adapters were trained on that encoder's posteriors, so they are not used.

**Density band.** The band belongs to the encoder: it is calibrated on that encoder's
training posteriors. [`configs/ms_gpt/msg-calibration.json`](../../../../../configs/ms_gpt/msg-calibration.json)
is the 4M-aug band. Its two thresholds make the mean predicted on-bit count 1.5 and 0.95
times the mean true count on the 194,119 MassSpecGym training spectra (paper
Appendix A.1.2). The run fails if the posteriors come from a different encoder than the
one the band records.

## Assets

From the [author-authorized mirrors](https://github.com/VIKI623/MS-GPT/issues/1):

```bash
python -m pip install -c environments/generation.txt -e '.[ms-gpt-base]'
hf download nielsr/MS-GPT-MLM --revision a139951c6cd2b6f31be9a53623a869610ee1d7ec \
  --local-dir checkpoints/ms-gpt/decoder
hf download nielsr/MS-GPT-SAFE-GPT-Base tokenizer.json \
  --revision 8d0560e93964a3344e724153c1bc83178d4a9d8f --local-dir checkpoints/ms-gpt
```

| Configuration key | Input |
|---|---|
| `decoder_checkpoint`, `decoder_config` | MS-GPT decoder weights and [`configs/ms_gpt/architecture.yaml`](../../../../../configs/ms_gpt/architecture.yaml) |
| `decoder_tokenizer` | the SAFE tokenizer JSON |
| `calibration` | the training density band of the encoder in use |
| `posterior`, `posterior_ids` | the released MIST *4M-aug* posterior cache, or `mist_checkpoint` + `compact_cache` to encode |

## Run

```bash
msms-models predict --config configs/ms_gpt/msg.yaml --device cuda:0
msms-models evaluate --protocol table5 --queries data/msg/queries.jsonl \
  --predictions results/ms-gpt-base/msg/predictions.jsonl --output results/ms-gpt-base/msg/metrics.json
```

A spectrum whose formula contains an element outside the decoder's vocabulary is
reported with status `unsupported_formula` and kept in the denominator. The
paper's Figure 2 decoders are character-level SMILES models queried the same way.
Their pretraining sets are in [`harrylau/ms-gpt-18`](https://huggingface.co/datasets/harrylau/ms-gpt-18).
