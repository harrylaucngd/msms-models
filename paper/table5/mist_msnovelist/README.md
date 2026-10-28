# Table 5 · MIST + MSNovelist

| | Accuracy | MCES | Tanimoto |
|---|---|---|---|
| Top-1 | 0.00% | 39.84 | 0.06 |
| Top-10 | 0.00% | 18.83 | 0.15 |

**What runs.** MIST, as released with DiffMS, encodes each spectrum's subformula tree into
a 4096-bit fingerprint posterior, 128 spectra per batch. A ctMSNovelist decoder trained on
Morgan radius-2/4096 fingerprints and formulas
([`models/de_novo/msnovelist`](../../../src/msms_models/models/de_novo/msnovelist/)) rounds
the posterior at 0.5 and decodes SMILES of the known formula by beam search (100 beams).
The best-scoring candidate per InChIKey-14 is kept, up to 10. Spectra are those of the
DiffMS evaluation, as in the [vDiffMS rows](../mist_vdiffms_molforge/).

**Training.** The decoder is trained on the 2.8M-molecule decoder pretraining set of
[DiffMS](https://github.com/coleygroup/DiffMS), 2D InChIKey-disjoint with the MassSpecGym
test and validation molecules. `build_database.py` writes the training database from the
SMILES; `train.py` trains with fold 0 as validation:

```bash
python -m msms_models.models.de_novo.msnovelist.build_database \
  --smiles fp2mol_train.txt --output data/msnovelist/train.db
python -m msms_models.models.de_novo.msnovelist.train --database data/msnovelist/train.db \
  --output checkpoints/msnovelist/run --fold 0 --epochs 20 --batch-size 256 --device cuda:0
cp checkpoints/msnovelist/run/w-<epoch with the lowest validation loss>.pth checkpoints/msnovelist/decoder.pth
```

**Inputs.** The official test queries from `paper/prepare_data.sh`; the DiffMS MIST
(`checkpoints/mist_vdiffms/mist_msg.pt`) and DiffMS subformula trees
(`data/paper/table5/mist_vdiffms/subformulae.jsonl`) from
[`harrylau/msms-models`](https://huggingface.co/harrylau/msms-models); the trained decoder.

## Run

From the repository root:

```bash
DEVICE=cuda:0 bash paper/table5/mist_msnovelist/run.sh
```

Set `OUT=...` to choose the output directory.
`run.sh` predicts and evaluates with the fixed `table5` protocol.
