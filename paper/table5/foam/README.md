# Table 5 · FOAM (DiffMS + PubChem)

| | Accuracy | MCES | Tanimoto |
|---|---|---|---|
| Top-1 | 1.50% | 12.21 | 0.35 |
| Top-10 | 10.28% | 6.06 | 0.53 |

**What runs.** FOAM's formula-constrained graph genetic algorithm
([`models/de_novo/foam`](../../../src/msms_models/models/de_novo/foam/)) evolves molecules
of the query's formula, scoring each by the entropy similarity between the observed
spectrum and the spectrum ICEBERG predicts for it, for up to 1,500 scoring calls per
spectrum (population 200, offspring 600, mutation rate 0.2). ICEBERG is the MassSpecGym
pair of [`simulators/iceberg`](../../../src/msms_models/simulators/iceberg/), also used by
FRIGID's refinement.

**Seeds.** The population starts from the seeds in `data/paper/table5/foam/seeds.jsonl`,
built by [`build_seeds.py`](build_seeds.py): the structures DiffMS proposes for each
spectrum and the PubChem structures with the query's formula, without PubChem structures
whose 2D InChIKey is a MassSpecGym test molecule. Seeds are tautomerized without
stereochemistry and deduplicated. Its two inputs:
- **DiffMS predictions** for the test spectra, as `{"id", "candidates"}` records, from
  [coleygroup/DiffMS](https://github.com/coleygroup/DiffMS) with its released MassSpecGym
  checkpoint.
- **PubChem's `CID-SMILES`** listing,
  from <https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz>.

**Inputs.** The official test queries from `paper/prepare_data.sh`, the ICEBERG pair
(`checkpoints/iceberg/msg_{generator,intensity}.ckpt`) from
[`harrylau/msms-models`](https://huggingface.co/harrylau/msms-models), and the seeds.

## Run

From the repository root:

```bash
python paper/table5/foam/build_seeds.py --queries data/msg/queries.jsonl \
  --diffms <diffms predictions.jsonl> --pubchem CID-SMILES.gz \
  --output data/paper/table5/foam/seeds.jsonl
DEVICE=cuda:0 bash paper/table5/foam/run.sh
```

Set `OUT=...` to choose the output directory.
`run.sh` predicts and evaluates with the fixed `table5` protocol.
