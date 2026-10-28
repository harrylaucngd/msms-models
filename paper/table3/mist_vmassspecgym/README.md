# Table 3 · MIST vMassSpecGym (no augmentation)

| Hit@1 | Hit@5 | Hit@20 | MCES@1 |
|---|---|---|---|
| 9.57% (8.88–10.30) | 22.11% (21.10–23.13) | 41.12% (39.98–42.34) | 12.75 (12.59–12.91) |

Intervals are 99.9% BCa over 20,000 spectrum-level resamples.

**What runs.** The MIST checkpoint released with MassSpecGym
([Zenodo 11580401](https://zenodo.org/records/11580401), `mist_fp.ckpt`) encodes each
spectrum's subformula tree with the MistNet of
[`encoders/mist/diffms.py`](../../../src/msms_models/encoders/mist/diffms.py)
(`mist_architecture: massspecgym`: 3 peak-attention layers, 5 refinement layers), 128
spectra per batch. Candidates are ranked by cosine similarity between the posterior and
their Morgan radius-2/4096 fingerprint, as MassSpecGym's `FromDictRetrieval` does.

**Inputs.** The official test queries and formula candidates from `paper/prepare_data.sh`;
the checkpoint (`checkpoints/mist_vmassspecgym/mist_fp.ckpt`) and the subformula trees of
the MassSpecGym MIST data (`data/paper/table3/mist_vmassspecgym/subformulae.jsonl`) from
[`harrylau/msms-models`](https://huggingface.co/harrylau/msms-models).

## Run

From the repository root:

```bash
bash paper/table3/mist_vmassspecgym/run.sh
```

Set `DEVICE=cuda:0` for GPU inference and `OUT=...` to choose the output directory.
`run.sh` predicts and evaluates with the fixed `table3` protocol.
