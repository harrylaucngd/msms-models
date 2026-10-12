# Model and data assets

Weights and large datasets are not stored in Git. The repository's model code
can load user-supplied assets immediately. Archival URLs for the exact benchmark
bundles are **pending**; the catalog's null links are placeholders, not completed
uploads. Do not substitute a similarly named checkpoint and assume benchmark
numbers will agree.

| Bundle | Required contents |
|---|---|
| MSG / CANOPUS inputs | Ordered query manifests, split metadata and prepared spectrum inputs; compact MIST feature caches where needed |
| MSG retrieval pools | Full ordered standard mass and bonus formula candidate libraries, keyed by query ID |
| Sub-formula NN | Portable train-only index with arrays, vocabulary, frozen parameters and validation selection manifest |
| MIST no augmentation / augmentation | Separate MSG and CANOPUS checkpoint files; each identified by dataset and training recipe |
| DiffMS | Complete fine-tuned model checkpoint, paired with the matching [model JSON](../model_configs/) |
| MolForge | Decoder checkpoint, source/target SentencePiece models and the encoder's fingerprint threshold |
| MS-GPT | SMILES decoder checkpoint, architecture YAML and a train-derived density-band JSON tied to the encoder |
| FRIGID | DLM checkpoint, SAFE tokenizer JSON, dataset-specific length predictor and feature-list JSON |
| ICEBERG | Paired generator and intensity checkpoints; MSG and CANOPUS pairs have distinct instrument vocabularies |
| GLACIER | Dataset-specific checkpoint matching the public MIST compatibility architecture |
| FOAM seeds | Explicit prediction/database seed lists, keyed by query, with a source manifest |

Suggested local layout:

```text
artifacts/
  mist/msg_noaug.ckpt
  mist/msg_4m_aug.ckpt
  diffms/msg_full.ckpt
  molforge/decoder.pth
  molforge/source.model
  molforge/target.model
  ms_gpt/decoder.pt
  ms_gpt/model.yaml
  ms_gpt/msg_density_band.json
  frigid/decoder.ckpt
  frigid/tokenizer.json
  frigid/msg_length.joblib
  frigid/msg_length_features.json
  iceberg/msg_generator.ckpt
  iceberg/msg_intensity.ckpt
  glacier/msg.ckpt
  subformula-msg/
data/
  queries.jsonl
  compact/msg/test/
  candidates_mass.jsonl
  candidates_formula.jsonl
  foam_seeds.jsonl
  foam_seed_manifest.json
```

Source resources are available from
[MassSpecGym](https://huggingface.co/datasets/roman-bushuiev/MassSpecGym),
[public MIST](https://github.com/coleygroup/MIST), and the
[upstream model repositories](references.md). Preserve their dataset/weight
licenses and split assignments. The compact MIST cache and source HDF5 archive
represent related data at different processing stages; inference needs the
compact features, not both archives.

## Small metadata examples

An MS-GPT architecture YAML contains only model architecture fields:

```yaml
model:
  d_model: 768
  n_layers: 12
  n_heads: 12
  fp_bits: 4096
  fp_encoder_layers: 2
  cross_attn_every_n: 2
  max_seq_len: 256
```

Its density-band JSON requires `benchmark` (`massspecgym` or `nplib1`),
`calibration_split: "train"`, `threshold_low` and `threshold_high`. Retain the
encoder identity, density targets and calibration sample count as provenance.
The endpoints must come from that encoder's training-split posterior calibration;
they are not universal constants.

When publishing a bundle, materialize external array references, retain IDs and
split metadata, remove machine-specific paths from manifests, and test it after
moving it to a different directory. Include the original model/data licenses and
state the exact recipe. Finally replace the null URLs in
[the artifact catalog](../src/msms_models/artifacts.json) with permanent archive
links and update these instructions. A code release does not itself archive the
weights or data.
