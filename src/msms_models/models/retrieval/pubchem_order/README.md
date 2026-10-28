# PubChem default ranking

A spectrum-blind baseline (paper Appendix A.4, method 4). For each query formula, PubChem
PUG-REST (`compound/fastformula`) lists the matching compounds in its default order, which
favours well-characterized molecules. [`build.py`](build.py) saves those listings to a JSONL
file of `{"formula", "cids", "inchikey14"}` rows. [`inference.py`](inference.py) ranks each
candidate by the first position of its InChIKey-14 in the listing of the query formula.
Unlisted candidates follow in a seeded random order, because the official pools list the true
molecule first. No spectrum is read.

## Assets

| Configuration key | Input |
|---|---|
| `pubchem_order` | The listing written by `build.py` |
| `candidates` | ID-keyed formula candidate pool |

**Parameters.** None. The configuration's `seed` sets the order of unlisted candidates.

The listing is built locally and needs internet access: about one request per formula, at most
five per second. An interrupted build resumes. PubChem's order changes over time, so a new
listing can rank differently.

## Run

From the repository root, after `bash paper/prepare_data.sh`:

```bash
bash paper/table3/pubchem_order/run.sh
```

`run.sh` builds the listing first if it is missing:

```bash
python -m msms_models.models.retrieval.pubchem_order.build \
  --queries data/msg/queries.jsonl --output data/paper/table3/pubchem_order.jsonl
```

## Source

Implemented here for the paper, on [PubChem PUG-REST](https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest).
