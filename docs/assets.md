# External resources

This repository does not host model weights or duplicate benchmark datasets.
Reuse the existing MIST Zenodo release and the model authors' published assets.
[The resource catalog](../src/msms_models/artifacts.json) distinguishes existing
MIST resources, official upstream downloads and generated inputs. For MIST assets,
follow the release links maintained by coleygroup/MIST; catalog entries with a
null `record_url` do not yet pin a specific Zenodo record.

| Resource | Provider / preparation |
|---|---|
| MSG/CANOPUS source spectra and compact features | Existing [coleygroup/MIST](https://github.com/coleygroup/MIST) Zenodo resources |
| MIST noaug/aug and GLACIER checkpoints | The same existing MIST release; reuse the original files |
| MSG spectra, splits and candidate pools | [Official MassSpecGym data](https://huggingface.co/datasets/roman-bushuiev/MassSpecGym) |
| MolForge decoder and tokenizers | [Original MolForge](https://github.com/knu-lcbc/MolForge), or a compatible user-supplied checkpoint |
| MS-GPT decoder, SAFE tokenizer and adapters | [Official MS-GPT](https://github.com/VIKI623/MS-GPT), which links the [released asset bundle](https://pan.sjtu.edu.cn/web/share/8f63823e4d82280636e537d1c789128d) |
| FRIGID decoder, tokenizer and length model | [Official FRIGID](https://github.com/coleygroup/FRIGID); obtain compatible assets or follow its length-model training instructions |
| ICEBERG generator/intensity pair | [Official ms-pred](https://github.com/coleygroup/ms-pred); use the matching MSG/CANOPUS pair |
| NN index | Build locally from the official training/validation splits with `fit-nn` |
| MIST posterior caches | Generate locally from existing features and encoder weights |
| Density calibration, FOAM seeds and manifests | Prepare locally for the selected encoder and experiment; record their provenance |

Training MIST from scratch or using synthetic augmentation is handled upstream.
The large synthetic training caches are not required for ordinary inference.
Source spectra and compact features serve different preparation stages; download
only those needed by your selected workflow.

## Local file layout

The configurations use the following layout; editing their paths is also supported:

```text
checkpoints/
  mist_msg_noaug.ckpt
  mist_msg_aug.ckpt
  mist_canopus_noaug.ckpt
  mist_canopus_aug.ckpt
  glacier_msg.ckpt
  molforge/{decoder.pth,source.model,target.model}
  ms-gpt/decoder.pt
  ms-gpt/tokenizer.json
  ms-gpt/adapters/{msg,canopus}/
  frigid/{decoder.ckpt,tokenizer.json}
  frigid/{msg,canopus}_length.joblib
  frigid/{msg,canopus}_length_features.json
  iceberg/{msg,canopus}_{generator,intensity}.ckpt
data/
  compact/{msg,canopus}/{train,val,test}/
  {msg,canopus}/queries.jsonl
  msg/candidates_{mass,formula}.jsonl
  {msg,canopus}/{foam_seeds.jsonl,foam_seed_manifest.json}
  indices/subformula-msg/
  calibration/ms-gpt/{msg,canopus}.json
```

These paths are the download destinations expected by the example configurations.
The MIST release filenames are retained. Tokenizers and LoRA adapters are also external model assets; none
are copied into this repository. The decoder architecture YAMLs are small text
configurations included beside the model's runnable configurations.

## MS-GPT base and full model

The official release provides one shared conditional SAFE decoder and
benchmark-specific posterior-aligned LoRA adapters. Both model IDs use density
calibration and group querying, normally 20 queries × 5 samples:

- `ms-gpt-base`: load the conditional decoder without an adapter.
- `ms-gpt`: load that decoder plus a matching trained adapter.

A PEFT adapter directory must contain `adapter_config.json` and
`adapter_model.safetensors` or `adapter_model.bin`. Configuration alone is not
an adapter. Available `training_config.json` provenance is checked against the
benchmark and calibration. The loader validates the complete adapter tensor set
before merging; it never silently runs the base model when an adapter is missing.

The density-band JSON requires `benchmark` (`massspecgym` or `nplib1`),
`calibration_split: "train"`, `threshold_low` and `threshold_high`. Record the
encoder identity and calibration sample count as well. An adapter trained with
one MIST encoder/band is not automatically a valid replacement for another.
Use each official adapter with its corresponding encoder and training calibration.
To change the encoder, follow the official MS-GPT training procedure to produce
a matching adapter; changing only the encoder path does not preserve the paper's
model configuration.
