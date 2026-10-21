# Configuration

Configurations mirror method ownership. Dataset names select the inputs; suffixes
select named variants of the same method.

```text
configs/
  nearest_neighbor/msg.yaml
  mist/{msg-noaug,msg-4m-aug}.yaml
  iceberg/msg.yaml
  glacier/msg.yaml
  molforge/{msg,canopus}.yaml
  ms_gpt/{msg,msg-base,canopus,canopus-base,architecture}.yaml
  frigid/{msg,msg-base,canopus,canopus-base}.yaml
  foam/{msg,canopus}.yaml
```

`msms-models models` includes configuration locations. In Python,
`get_model("frigid-base").config_for("msg")` returns
`configs/frigid/msg-base.yaml`. Architecture YAMLs are construction assets, not
runnable configurations. Base/full MS-GPT share one architecture file.

Paths are relative to the YAML file. `../../checkpoints/` resolves to the checkout's
checkpoint directory. Set `python` to select a model environment, or omit it to use
the current interpreter. `device`, `workers` and `seed` are explicit.

```bash
msms-models doctor --config configs/ms_gpt/msg-base.yaml
msms-models infer --config configs/ms_gpt/msg-base.yaml --device cuda:0
msms-models infer --config configs/ms_gpt/msg.yaml --device cuda:0
```

`--queries` and `--output` override input/output paths without editing a template.
`--dry-run` validates a run without loading weights. A new run requires a fresh
output directory. `smoke --config ... --output ... --limit 2` selects the first two
queries, keeps the configured inference budget and also evaluates predictions when
target structures are supplied.

Every asset must match the selected method and dataset. MolForge/FRIGID configurations pin the prepared encoder hash and its
validation-calibrated threshold. Changing an encoder requires recalibrating its fingerprint threshold. MS-GPT's calibration and adapter must be trained with the same MIST encoder.
Changing only the encoder path does not preserve the original model configuration.

Pin known checkpoint identities with an optional map:

```yaml
asset_sha256:
  decoder_checkpoint: YOUR_VERIFIED_SHA256
```

Replace the placeholder with the file's SHA-256. A mismatched file or a directory
in this map is rejected. Unpinned files are still hashed in each run's `inputs.json`.

FRIGID's `num_rounds` counts the initial generation round. Ten refinement steps
therefore require eleven total rounds; `frigid-base` executes initial generation
only. See the method guide for proposal budgets and the results note for archived
experiment settings.
