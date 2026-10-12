# Configuration layout

Every model has a directory under `configs/<model>/`. Runnable configurations
are named by dataset: `msg.yaml` and, for the de novo models, `canopus.yaml`.
Molecular retrieval configurations currently cover MSG only.

```text
configs/
  mist-noaug/msg.yaml
  mist-4m-aug/msg.yaml
  subformula-nn/msg.yaml
  iceberg/msg.yaml
  glacier/msg.yaml
  molforge/{msg,canopus}.yaml
  ms-gpt-base/{msg,canopus,architecture}.yaml
  ms-gpt/{msg,canopus,architecture}.yaml
  frigid-base/{msg,canopus}.yaml
  frigid/{msg,canopus}.yaml
  foam/{msg,canopus}.yaml
```

`architecture.yaml` is a checkpoint construction asset, not a runnable inference
configuration. Only models whose checkpoint format needs external architecture
information have this file. Other loaders read that information from checkpoint
metadata or use their specified decoder architecture. There is no separate
top-level model-configuration directory.

Paths are relative to the YAML file that contains them. In a model's `msg.yaml`,
`../../checkpoints/` means the checkout's checkpoint directory, while
`architecture.yaml` means the architecture file beside that run configuration.
Downloaded weights stay outside Git. Use a matching Python environment by
setting the optional `python` field; otherwise the current interpreter is used.

```bash
msms-models doctor --config configs/ms-gpt-base/msg.yaml
msms-models infer --config configs/ms-gpt-base/msg.yaml
msms-models infer --config configs/ms-gpt/msg.yaml
```

`ms-gpt-base` and `ms-gpt` are independent model IDs. Both use the calibrated
query band; only `ms-gpt` requires `adapter_checkpoint`. Base rejects an adapter
asset, and full MS-GPT rejects a missing, incomplete or incompatible adapter.
The same separate-model convention applies to `frigid-base` and `frigid`.

The MS-GPT architecture follows the paper's official SAFE implementation.
Set `decoder_tokenizer` to its released tokenizer JSON. The decoder, tokenizer,
posterior-aligned adapter, MIST encoder and training calibration must be compatible;
see the [asset guide](assets.md) for the required files.
