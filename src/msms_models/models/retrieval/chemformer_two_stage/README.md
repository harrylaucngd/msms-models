# Chemformer two-stage retrieval (GLMR)

GLMR is the authors' implementation of *Generative Language Model Retrieval*
(arXiv:2511.06259), row (2) of Table 3. It ranks candidates in two stages:

1. **Pre-retrieval.** A 6-layer spectral transformer embeds the 61 most intense
   peaks, sorted by m/z, and mean-pools them. It is trained contrastively against
   the frozen CLS embedding of the pretrained Chemformer encoder. Each candidate
   gets an L2-normalised CLS embedding, and the whole pool is ranked by cosine
   similarity.
2. **Generative re-ranking.** The spectrum's token states attend to the
   concatenated encoder states of the top `pre_retrieval_k = 40` candidates (one
   cross-attention layer). The Chemformer decoder then runs beam search (beam 5, up
   to 512 tokens) on the fused states. The 40 candidates are re-ranked by cosine
   similarity to the decoded molecule's CLS embedding.

`predict` returns every candidate: the re-ranked top 40, then the rest in stage-1
order. When the decoded string is empty, the stage-1 order is kept. `scores` holds the
stage-2 similarity for the first `reranked` entries and the stage-1 similarity after them.
On CUDA every forward pass runs under fp16 autocast.

| File | Contents (source in GLMR) |
|---|---|
| `tokenizer.py` | Chemformer's SMILES regex and 523-token BART vocabulary (`src/data/tokenizer.py`) |
| `model.py` | Chemformer encoder/decoder, spectral encoder, cross-fusion and both stage models (`src/models/*`) |
| `data.py` | datasets and batching (`src/data/*`, `src/evaluate/eval_full.py`) |
| `inference.py` | two-stage ranking (`benchmark_massspecgym_full/run_full_with_ci.py`) |
| `train.py` | stage-1 training, top-K precomputation and stage-2 training (`src/train_pre_retrieval.py`, `src/precompute_candidates.py`, `src/train_generative.py`) |

The Chemformer architecture is implemented in `model.py` to load the Chemformer
checkpoint without PyTorch Lightning or DeepSpeed; the callback and loss-scaler objects
pickled beside the weights are replaced by placeholders.

## Assets

| Configuration key | Input |
|---|---|
| `chemformer_checkpoint` | Chemformer pretrained BART `step=1000000.ckpt` (task `mask_aug`, d_model 512, 6+6 layers, vocabulary 523) from [MolecularAI/Chemformer](https://github.com/MolecularAI/Chemformer), Apache-2.0 |
| `pre_retrieval_checkpoint` | Stage-1 `best_model.pt` (`spec_encoder_state_dict`, training `config`) |
| `generative_checkpoint` | Stage-2 `best_gen_model.pt` (`cross_fusion_state_dict`, `decoder_state_dict`, `config`) |
| `candidates` | Ordered candidate pool per query ID (`prepare-candidates` output) |

**Parameters.** `max_peaks` (61), `pre_retrieval_k` (40), `beam_size` (5), `max_gen_len`
(512), `eval_batch_size` (32 queries per batch) and `candidate_batch_size` (512 candidates
per batch).

## Run

From the repository root, after `bash paper/prepare_data.sh`:

```bash
bash paper/table3/chemformer_two_stage/run.sh
```

`run.sh` evaluates the inflated and corrected rows; see
[the paper row](../../../../../paper/table3/chemformer_two_stage/README.md).

## Training

Each released checkpoint stores its training `config`, which is a valid input, and
`train.default_config(stage, variant)` writes the same configuration apart from file paths.

```bash
python -m msms_models.models.retrieval.chemformer_two_stage.train config --stage pre-retrieval \
  --variant corrected --mgf data/SpecBridge_MassSpecGym_dataset_ca.mgf \
  --candidates data/SpecBridge_MSGYM_candidates_ca.pkl --chemformer checkpoints/chemformer/step=1000000.ckpt \
  --save-dir runs/glmr/pre_retrieval --output runs/glmr/pre.yaml
python -m msms_models.models.retrieval.chemformer_two_stage.train pre-retrieval --config runs/glmr/pre.yaml
python -m msms_models.models.retrieval.chemformer_two_stage.train precompute --config runs/glmr/pre.yaml \
  --pre-retrieval-checkpoint runs/glmr/pre_retrieval/best_model.pt --output-dir runs/glmr/precomputed
python -m msms_models.models.retrieval.chemformer_two_stage.train config --stage generative \
  --variant corrected ... --pre-retrieval runs/glmr/pre_retrieval/best_model.pt \
  --precomputed-dir runs/glmr/precomputed --save-dir runs/glmr/generative --output runs/glmr/gen.yaml
python -m msms_models.models.retrieval.chemformer_two_stage.train generative --config runs/glmr/gen.yaml
```

Training reads an MGF file and a candidate pickle. The pickles are keyed by target SMILES,
and precomputed top-K files are keyed by the record's index within its split.

| Setting | Stage 1 (both rows) | Stage 2, inflated | Stage 2, corrected |
|---|---|---|---|
| Data | raw / `_ca` MGF | raw MGF + pickle | `_ca` MGF + pickle |
| Epochs, batch, AdamW lr / wd | 300, 256, 1e-4 / 0.1 | 30, 128, 1e-4 / 0.1 | 30, 128, 1e-4 / 0.1 |
| Validation every | 5 epochs | 1 epoch | 5 epochs |
| Gradient accumulation | none | 4 | 1 |
| Clipping (norm 1) | none | after `scaler.unscale_` | on loss-scaled gradients (`legacy_scaled_clip: true`) |
| Selected checkpoint (min. val. loss) | epoch 240 / 200 | epoch 4 | epoch 10 |

`training.legacy_scaled_clip: true` clips the loss-scaled gradients instead of the
unscaled ones; the corrected default configuration sets it. Stage 1 uses an
intensity-noised copy of each spectrum as the negative, with noise in ±30%, τ = 0.1 and
seed 42. Pass the device with `--device`.

## Source

[harrylaucngd/GLMR](https://github.com/harrylaucngd/GLMR) at `c6327e2`; the GLMR
repository has no LICENSE file. Chemformer and its pretrained weights are Apache-2.0
([MolecularAI/Chemformer](https://github.com/MolecularAI/Chemformer)); no Chemformer source
code is copied. See the [third-party notices](../../../../../THIRD_PARTY_NOTICES.md).
