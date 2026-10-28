# MS-GPT

MS-GPT's SAFE molecule language model is queried with MIST *4M-aug* fingerprint
posteriors. For each spectrum the posterior is thresholded at 20 points of a
density-calibrated band to give 20 binary fingerprint queries. Five molecules are
sampled per query, filtered by the known formula, grouped by InChIKey-14 and ranked
by frequency, then by Tanimoto to the queries (paper Appendix A.1.2). A spectrum whose
formula contains an element outside the decoder's vocabulary is reported with status
`unsupported_formula` and kept in the denominator.

- [`inference.py`](inference.py) loads the decoder strictly and runs the querying and ranking.
- `vendor/` holds MS-GPT's decoder (`model.py`), density queries (`queries.py`) and ranking (`ranking.py`).

The density band belongs to the encoder: it is calibrated on that encoder's training
posteriors. [`paper/figure2/ms_gpt/calibration.json`](../../../../../paper/figure2/ms_gpt/calibration.json)
is the MIST *4M-aug* band. Its two thresholds make the mean predicted on-bit count 1.5 and
0.95 times the mean true count on the 194,119 MassSpecGym training spectra. MS-GPT's own
MIST encoder and its posterior-aligned LoRA adapters are not used (paper §6.1).

## Assets

| Configuration key | Input |
|---|---|
| `decoder_checkpoint`, `decoder_config` | MS-GPT decoder weights and [`paper/figure2/ms_gpt/architecture.yaml`](../../../../../paper/figure2/ms_gpt/architecture.yaml) |
| `decoder_tokenizer` | the SAFE tokenizer JSON |
| `calibration` | the training density band of the encoder |
| `posterior`, `posterior_ids` | the MIST *4M-aug* posterior cache |

**Parameters.** `num_queries` (20) is the number of binary queries per spectrum, and
`samples_per_query` (5) the molecules sampled per query. Sampling uses `temperature` (1.0),
`top_k` (50), `top_p` (0.95) and `max_length` (the architecture's `max_seq_len`, 256).

## Run

From the repository root, after `bash paper/prepare_data.sh`:

```bash
bash paper/figure2/ms_gpt/run.sh
```

`run.sh` downloads the decoder and tokenizer from the
[author-authorized mirrors](https://github.com/VIKI623/MS-GPT/issues/1), then predicts and
evaluates with [`paper/figure2/ms_gpt/config.yaml`](../../../../../paper/figure2/ms_gpt/config.yaml):

```bash
hf download nielsr/MS-GPT-MLM --revision a139951c6cd2b6f31be9a53623a869610ee1d7ec \
  --local-dir checkpoints/ms-gpt/decoder
hf download nielsr/MS-GPT-SAFE-GPT-Base tokenizer.json \
  --revision 8d0560e93964a3344e724153c1bc83178d4a9d8f --local-dir checkpoints/ms-gpt
```

The paper's Figure 2 decoders are character-level SMILES models queried the same way.
Their pretraining sets are in [`harrylau/ms-gpt-18`](https://huggingface.co/datasets/harrylau/ms-gpt-18).

## Source

[VIKI623/MS-GPT](https://github.com/VIKI623/MS-GPT) at `f2c844e61f07`, Apache-2.0; see
[`vendor/LICENSE.txt`](vendor/LICENSE.txt) and the
[third-party notices](../../../../../THIRD_PARTY_NOTICES.md).
