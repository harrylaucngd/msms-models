# Figure 2 · Decoder querying

Figure 2 queries molecule decoders with MIST *4M-aug* posteriors: each spectrum's posterior
is thresholded at 20 points of a density-calibrated band, giving 20 binary fingerprint
queries. Five molecules are sampled per query, filtered by the known formula, grouped by
InChIKey-14 and ranked by frequency, then by Tanimoto to the queries (Appendix A.1.2).
This folder runs that procedure with the released MS-GPT SAFE decoder
([`models/de_novo/ms_gpt`](../../../src/msms_models/models/de_novo/ms_gpt/)). The
pretraining sets of the 18 Figure 2 decoders (2.5M, 10M and 50M parameters on six splits)
are in [`harrylau/ms-gpt-18`](https://huggingface.co/datasets/harrylau/ms-gpt-18).

| File | Contents |
|---|---|
| [`config.yaml`](config.yaml) | 20 queries, 5 samples each, temperature 1.0, top-k 50, top-p 0.95 |
| [`architecture.yaml`](architecture.yaml) | the MS-GPT SAFE decoder |
| [`calibration.json`](calibration.json) | the MIST *4M-aug* density band: thresholds giving 1.5 and 0.95 times the true on-bit count on the 194,119 training spectra |

**Inputs.** The official test queries from `paper/prepare_data.sh`, the MIST *4M-aug*
posteriors from [`harrylau/msms-models`](https://huggingface.co/harrylau/msms-models), and
the MS-GPT decoder and tokenizer, which `run.sh` downloads from the
[author-authorized mirrors](https://github.com/VIKI623/MS-GPT/issues/1).

## Run

From the repository root:

```bash
DEVICE=cuda:0 bash paper/figure2/ms_gpt/run.sh
```

Set `OUT=...` to choose the output directory.
`run.sh` predicts and evaluates with the fixed `table5` protocol.
