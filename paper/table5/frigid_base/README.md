# Table 5 · FRIGID-base

| | Acc@1 | MCES@1 | Tanimoto@1 | Acc@10 | MCES@10 | Tanimoto@10 |
|---|---|---|---|---|---|---|
| FRIGID-base | 8.21% | 13.16 | 0.38 | 9.82% | 11.65 | 0.43 |

**Source.** [`models/de_novo/frigid`](../../../src/msms_models/models/de_novo/frigid/) ([coleygroup/FRIGID](https://github.com/coleygroup/FRIGID) DLM).

**What runs.** The first FRIGID generation round: 128 formula- and fingerprint-conditioned proposals per spectrum with the original NGBoost length predictor, ranked natively.

**Inputs.** Official MassSpecGym test queries with known formulas; the released MIST 4M-aug Table 5 posterior cache; FRIGID DLM, SAFE tokenizer and MSG length predictor.

**Environment.** `pip install -c environments/generation.txt -e '.[frigid,mces]'`.

## Run

From the repository root:

```bash
bash paper/table5/frigid_base/run.sh
```

Set `DEVICE=cuda:0` for GPU inference and `OUT=...` to choose the output directory.
`run.sh` predicts and evaluates with the fixed `table5` protocol.
