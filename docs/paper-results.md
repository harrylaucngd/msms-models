# MSG in the Wild: method and result provenance

This note maps Tables 3 and 5 in the author manuscript inspected on 2026-10-20
to the library. A manuscript includes literature results, author reruns and
diagnostic experiments; these have different implementation and asset provenance.
All integrated methods use the standard `infer` and `evaluate` interfaces.

The two local reimplementations called out in those tables are Nearest Neighbor
and MolForge. Their complete local inference source is in
[`methods/nearest_neighbor`](../src/msms_models/methods/nearest_neighbor/) and
[`methods/molforge`](../src/msms_models/methods/molforge/). MIST encoding is shared
through [`methods/mist/encoder.py`](../src/msms_models/methods/mist/encoder.py).

**MSNovelist:** the MIST + MSNovelist numbers were quoted from the DiffMS
comparison. Local historical experiment notes attribute that retrained baseline
to DiffMS and record that its matching checkpoint was unavailable. This repository
does not contain an MSNovelist reimplementation. The [original MSNovelist source](https://github.com/meowcat/MSNovelist)
and [DiffMS comparison](https://github.com/coleygroup/DiffMS) must not be described
as an implementation provided by msms-models.

## Table 3

| Manuscript method | Result origin | Source / configuration |
|---|---|---|
| Random | Cited result | [Upstream source](https://github.com/pluskal-lab/MassSpecGym); not integrated here |
| Nearest Neighbor | Local reimplementation | [nearest_neighbor source](../src/msms_models/methods/nearest_neighbor/), [`subformula-nn`](../configs/nearest_neighbor/msg.yaml) |
| DeepSets | Cited result | [Upstream source](https://github.com/pluskal-lab/MassSpecGym); not integrated here |
| Fingerprint FFN | Cited result | [Upstream source](https://github.com/pluskal-lab/MassSpecGym); not integrated here |
| DeepSets + Fourier features | Cited result | [Upstream source](https://github.com/pluskal-lab/MassSpecGym); not integrated here |
| MIST vMassSpecGym (no augmentation) | Cited result | Historical MassSpecGym MIST assets; current `mist-noaug` is a different checkpoint |
| JESTR | Cited result | [Upstream source](https://doi.org/10.1093/bioinformatics/btaf354); not integrated here |
| MVP | Cited result | [Upstream source](https://doi.org/10.1021/acs.analchem.5c05675); not integrated here |
| FLARE | Cited result | [Upstream source](https://doi.org/10.64898/2026.01.27.702086); not integrated here |
| MIST 4M-aug | Author rerun | [mist source](../src/msms_models/methods/mist/), [`mist-4m-aug`](../configs/mist/msg-4m-aug.yaml) |
| ICEBERG | Author rerun | [iceberg source](../src/msms_models/methods/iceberg/), [`iceberg`](../configs/iceberg/msg.yaml) |
| GLACIER | Author rerun | [glacier source](../src/msms_models/methods/glacier/), [`glacier`](../configs/glacier/msg.yaml) |
| DreaMS + ChemBERTa Alignment (inflated) | Diagnostic experiment | Original experimental implementation/assets not yet integrated or released |
| DreaMS + ChemBERTa Alignment (corrected) | Author experiment | Original experimental implementation/assets not yet integrated or released |
| Chemformer Two-Stage Retrieval (inflated) | Diagnostic experiment | Original experimental implementation/assets not yet integrated or released |
| Chemformer Two-Stage Retrieval (corrected) | Author experiment | Original experimental implementation/assets not yet integrated or released |
| ChemBERTa SMILES Classifier (inflated) | Diagnostic experiment | Original experimental implementation/assets not yet integrated or released |
| ChemBERTa SMILES Classifier (corrected) | Author experiment | Original experimental implementation/assets not yet integrated or released |
| PubChem Default Ranking | Author experiment | Original experimental implementation/assets not yet integrated or released |
| Ranking by Number of Chiral Atoms | Author experiment | Original experimental implementation/assets not yet integrated or released |

## Table 5

| Manuscript method | Result origin | Source / configuration |
|---|---|---|
| SMILES Transformer | Cited result | [Upstream source](https://github.com/pluskal-lab/MassSpecGym); not integrated here |
| SELFIES Transformer | Cited result | [Upstream source](https://github.com/pluskal-lab/MassSpecGym); not integrated here |
| Random generation | Cited result | [Upstream source](https://github.com/pluskal-lab/MassSpecGym); not integrated here |
| MIST + MSNovelist | Cited result | [Upstream source](https://github.com/meowcat/MSNovelist); not integrated here |
| Spec2Mol | Cited result | [Upstream source](https://github.com/KavrakiLab/Spec2Mol); not integrated here |
| MIST + Neuraldecipher | Cited result | [Upstream source](https://github.com/bayer-science-for-a-better-life/neuraldecipher); not integrated here |
| MADGEN | Cited result | [Upstream source](https://openreview.net/forum?id=78tc3EiUrN); not integrated here |
| MS-BART | Cited result | [Upstream source](https://github.com/OpenDFM/MS-BART); not integrated here |
| DiffMS | Cited result | [Upstream source](https://github.com/coleygroup/DiffMS); not integrated here |
| FOAM (DiffMS + PubChem) | Cited result | [foam source](../src/msms_models/methods/foam/), [`foam`](../configs/foam/msg.yaml); cited numbers are not certified by the current configuration |
| MSAnchor | Cited result | [Upstream source](https://ojs.aaai.org/index.php/AAAI/article/view/37064); not integrated here |
| MIST + MolForge (corrected) | Local reimplementation | [molforge source](../src/msms_models/methods/molforge/), [`molforge`](../configs/molforge/msg.yaml) |
| MIST + MolForge (inflated, batch size 24) | Diagnostic experiment | Original experimental implementation/assets not yet integrated or released |
| FRIGID-base | Author rerun | [frigid source](../src/msms_models/methods/frigid/), [`frigid-base`](../configs/frigid/msg-base.yaml) |
| FRIGID (10 refinements) | Author rerun | [frigid source](../src/msms_models/methods/frigid/), [`frigid`](../configs/frigid/msg.yaml) |
| FRIGID with MIST vFRIGID | Diagnostic experiment | Original experimental implementation/assets not yet integrated or released |

## Running an integrated method

Use the data, assets and commands in its method guide. For example, MolForge:

```bash
msms-models doctor --config configs/molforge/msg.yaml
msms-models infer --config configs/molforge/msg.yaml --device cuda:0
python -m pip install -e '.[mces]'
msms-models evaluate --task de-novo --queries data/msg/queries.jsonl \
  --predictions results/molforge/msg/predictions.jsonl \
  --output results/molforge/msg/metrics.json --mces --mces-cache results/mces.sqlite
```

Use all 17,556 MSG test spectra for the author reruns, preserving missing outputs
and raw beam positions. The standard evaluator records its settings. Do not use
a smoke-test metric, a new candidate pool or a substitute checkpoint as the
reported experiment.

## Experimental identities and remaining release work

The current normal configurations document valid method inputs. Matching a paper
row additionally requires its exact asset versions and preprocessing. The following
identities were recovered from experiment records; they do not create download URLs.

| Component | Recorded SHA-256 / setting |
|---|---|
| Corrected MSG MIST encoder used for MolForge/FRIGID | `548c1301d9093c732f4a79666e118fb87924de97576fd7c5bd3f17a104e659b6` |
| Portable MSG MIST EMA release (declares the source above) | `9f251d772f68315ba6fe2dc75a3b2e1ec477d3f41ffbeaf5143749157b67b177` |
| FRIGID decoder | `b6177c2d43448380aba80ff41c01461ea34ca2ca93b213986954c5afb7f0f457` |
| MSG MolForge fingerprint threshold | `0.18791014701128006`, ten raw beams, formula filter disabled |
| FRIGID MSG refinement budget | Ten refinement transitions = eleven rounds, including initial generation; 128 proposals per round |

The normal MSG MolForge/FRIGID configurations pin the portable encoder file hash,
not the original training checkpoint hash. The first two full MIST encodings
were verified to give identical thresholded bits to the archived predictions.
This is a subset parity check, not proof of every archived inference setting.

The remaining gaps are material:

- Public MIST compact-cache/checkpoint uploads have null URLs and checksums.
  A fresh user cannot yet obtain every necessary asset from the documented providers.
- The exact MolForge decoder/tokenizer triplet is pinned by hash and tested
  locally, but its public download locations remain unverified. FRIGID's public
  archive omits the original length predictors and tokenizer.
- ICEBERG/GLACIER full-table runs require their frozen imputed collision energies,
  prepared peaks and exact checkpoints. The official TSV alone is insufficient.
- The Table 3 MIST result needs the frozen candidate pool and tie convention.
  A historical audit differs slightly from the current manuscript; that discrepancy
  remains unresolved and is not described as exact reproduction.
- FOAM's cited DiffMS + PubChem result requires the original seed pool and seed
  provenance; the implementation cannot reconstruct it from target molecules.
- The separate diagnostic/author experiments listed above still need their
  original source and assets before they can be supported library methods.

Archived full-test MolForge predictions were independently evaluated through the
common evaluator: Acc@1 **3.29%**, Acc@10 **5.65%**, Tanimoto@1 **0.31**, and
Tanimoto@10 **0.40**, matching the printed precision. This checks aggregation of
17,556 saved predictions, not regenerated encoder/decoder predictions. MCES and
confidence intervals were not recomputed in that check.

See [validation](validation.md) for implementation tests and [assets](assets.md)
for public resource availability.
