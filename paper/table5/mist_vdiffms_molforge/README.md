# Tables 4 and 5 · MIST vDiffMS + MolForge (diagnostic)

| | Acc@1 | MCES@1 | Tanimoto@1 | Acc@10 | MCES@10 | Tanimoto@10 | Encoder Tanimoto |
|---|---|---|---|---|---|---|---|
| batch size 1 | 10.73% | 22.15 | 0.37 | 14.48% | 17.88 | 0.41 | 0.457 |
| batch size 24 (inflated) | 31.75% | 12.30 | 0.68 | 40.55% | 9.80 | 0.74 | 0.812 |

These rows show how the MIST padding-mask bug (paper §6.1.1) inflated the original
MIST + MolForge result. Upstream attention adds the raw boolean padding mask
(`attn += attn_mask`), so padded positions get +1 instead of −inf. At batch size 1 there is
no padding and the bug is dormant. At batch size 24 every spectrum shorter than the longest
in its batch is padded, and the posteriors of the DiffMS checkpoint are inflated.

**Source.** [`encoders/mist/diffms.py`](../../../src/msms_models/encoders/mist/diffms.py),
the MIST released with [DiffMS](https://github.com/coleygroup/DiffMS)
([vendor notes](../../../src/msms_models/encoders/mist/vendor/README.md)). It feeds
[`models/de_novo/molforge`](../../../src/msms_models/models/de_novo/molforge/)
([knu-lcbc/MolForge](https://github.com/knu-lcbc/MolForge), CC BY-NC-SA 4.0).

**What runs.** The DiffMS MIST encodes each spectrum's subformula tree. Spectra are encoded
in cohort order, `encoder_batch_size` consecutive spectra per pass, as the original
`DataLoader(shuffle=False)` did. The 4096-bit posterior is thresholded at 0.172 and MolForge
decodes ten beams.

| Config | `attention_padding` | `encoder_batch_size` |
|---|---|---|
| [`config-bs1.yaml`](config-bs1.yaml) | `masked` (−inf on padding) | 1 |
| [`config-bs24.yaml`](config-bs24.yaml) | `additive-bool` (historical +1) | 24 |

The encoder Tanimoto is the mean Tanimoto between the thresholded posterior and the
target's Morgan fingerprint (radius 2, 4096 bits).

**Cohort.** DiffMS keeps a test spectrum only if its target contains nothing outside
{C, O, P, N, S, Cl, F, H}. That gives 17,082 of the 17,556 spectra, listed in DiffMS's order
in [`cohort_ids.txt`](cohort_ids.txt).

**Inputs.**
- **Queries:** the official test queries, restricted to the cohort by `run.sh`.
- **Subformula trees:** DiffMS's trees, `data/paper/table5/mist_vdiffms/subformulae.jsonl`.
- **Checkpoints:** the DiffMS MIST checkpoint (`checkpoints/mist_vdiffms/mist_msg.pt`, from [Zenodo 15122968](https://zenodo.org/records/15122968)) and the MolForge decoder and tokenizers of the `mist_molforge` row.

**Environment.** `pip install -c environments/generation.txt -e '.[molforge,mces]'`.

## Run

```bash
bash paper/table5/mist_vdiffms_molforge/run.sh
```

Set `DEVICE=cuda:0` and `OUT=...` as needed.
