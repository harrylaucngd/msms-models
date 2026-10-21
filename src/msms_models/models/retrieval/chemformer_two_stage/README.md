# Chemformer two-stage retrieval (GLMR)

This is a port of [harrylaucngd/GLMR](https://github.com/harrylaucngd/GLMR) at
`c6327e2`. GLMR is the authors' reimplementation of *Generative Language Model
Retrieval* (arXiv:2511.06259). It is row (2) of Table 3 in *MassSpecGym in the Wild*.
Ranking and training follow the original step by step.

1. **Pre-retrieval.** A 6-layer spectral transformer embeds the 61 most intense
   peaks, sorted by m/z, and mean-pools them. It is trained contrastively against
   the frozen CLS embedding of the pretrained Chemformer encoder. Each candidate
   gets an L2-normalised CLS embedding, and the whole pool is ranked by cosine
   similarity with `np.argsort(-similarity)`. That is NumPy's default quicksort,
   so ties fall in the original order.
2. **Generative re-ranking.** The spectrum's token states attend to the
   concatenated encoder states of the top `pre_retrieval_k = 40` candidates (one
   cross-attention layer). The Chemformer decoder then runs beam search (beam 5, up
   to 512 tokens) on the fused states. The 40 candidates are re-ranked by cosine
   similarity to the decoded molecule's CLS embedding.

`predict` returns every candidate: the re-ranked top 40, then the rest in stage-1
order. When the decoded string is empty, the stage-1 order is kept, as in the
original. Only the head of the list differs from the original's ranks, and only for
spectra whose target falls outside the top 40. GLMR gave all of those spectra rank
`len(pool)`, while here the target keeps its stage-1 position. Hit@1/5/20 and
MCES@1 are unaffected. `scores` holds the stage-2 similarity for the first
`reranked` entries and the stage-1 similarity after them.

| File | Origin |
|---|---|
| `tokenizer.py` | GLMR `src/data/tokenizer.py`: Chemformer's SMILES regex and 523-token BART vocabulary |
| `model.py` | GLMR `src/models/*`: Chemformer encoder/decoder (reimplemented to load the Chemformer checkpoint), spectral encoder, cross-fusion, both stage models |
| `data.py` | GLMR `src/data/*` and the batching helpers of `src/evaluate/eval_full.py` |
| `inference.py` | `evaluate_full_collect` of GLMR `benchmark_massspecgym_full/run_full_with_ci.py` |
| `train.py` | GLMR `src/train_pre_retrieval.py`, `src/precompute_candidates.py`, `src/train_generative.py` |

No Chemformer source code is copied, so there is no `vendor/` directory. The
Chemformer checkpoint is loaded without PyTorch Lightning or DeepSpeed: the
callback and loss-scaler objects pickled beside the weights are replaced by
placeholders. On CUDA every forward pass runs under fp16 autocast, as in the
original. On CPU it runs in fp32, which is what the original's
`autocast("cuda")` did there.

## Assets

| Configuration key | Input |
|---|---|
| `chemformer_checkpoint` | Chemformer pretrained BART `step=1000000.ckpt` (task `mask_aug`, d_model 512, 6+6 layers, vocabulary 523, Lightning 1.2.3) from [MolecularAI/Chemformer](https://github.com/MolecularAI/Chemformer), Apache-2.0 |
| `pre_retrieval_checkpoint` | Stage-1 `best_model.pt` (`spec_encoder_state_dict`, training `config`) |
| `generative_checkpoint` | Stage-2 `best_gen_model.pt` (`cross_fusion_state_dict`, `decoder_state_dict`, `config`) |
| `candidates` | Ordered candidate pool per query ID (`prepare-candidates` output) |

**Parameters.** Defaults are those of the original benchmark: `max_peaks` (61),
`pre_retrieval_k` (40), `beam_size` (5), `max_gen_len` (512), `eval_batch_size`
(32) and `candidate_batch_size` (512).

Embeddings and beam search are batched, and under fp16 the results can depend on
batch shape and hardware. Keep the original batching: queries in file order,
32 per batch, and candidates in sorted order, 512 per batch. The original encoded
every SMILES of its candidate file across all splits, while this port encodes only
the candidates of the queries it receives. The order of the candidate list is
preserved, and so is the true molecule's position in it (index 0 in both official
pools). That position matters because quicksort ties are broken by input position.

The original environment's PyTorch build has no sm_120 kernels, so it does not run on
RTX PRO 6000 (Blackwell) GPUs. See
[the paper row](../../../../../paper/table3/chemformer_two_stage/README.md).

## Run

```bash
python -m pip install -e '.[chemformer-two-stage,mces]'
bash paper/table3/chemformer_two_stage/run.sh
```

## Training

The configurations keep GLMR's layout. The `config` stored in each released
checkpoint is therefore a valid input, and `train.default_config(stage, variant)`
reproduces it apart from file paths.

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

Training uses the MGF and candidate pickles of the released runs. The pickles are
keyed by target SMILES, and precomputed top-K files are keyed by the record's
index within its split.

| Setting | Stage 1 (both rows) | Stage 2, inflated | Stage 2, corrected |
|---|---|---|---|
| Data | raw / `_ca` MGF | raw MGF + pickle | `_ca` MGF + pickle |
| Epochs, batch, AdamW lr / wd | 300, 256, 1e-4 / 0.1 | 30, 128, 1e-4 / 0.1 | 30, 128, 1e-4 / 0.1 |
| Validation every | 5 epochs | 1 epoch | 5 epochs |
| Gradient accumulation | none | 4 | 1 |
| Clipping (norm 1) | none | after `scaler.unscale_` | on loss-scaled gradients (`legacy_scaled_clip: true`) |
| Selected checkpoint (min. val. loss) | epoch 240 / 200 | epoch 4 | epoch 10 |

The corrected stage-2 checkpoint was trained before GLMR added gradient
accumulation, with a loop that clipped the scaled gradients.
`training.legacy_scaled_clip: true` restores that loop, and the corrected default
configuration sets it. Stage 1 uses an intensity-noised copy of each spectrum as the
negative, with noise in ±30%, τ = 0.1 and seed 42. The noise is drawn from the
per-worker NumPy RNG that PyTorch seeds, as in the original. The port drops the
optional Weights & Biases logging and the automatic CUDA device choice (pass
`--device`).

## Licenses

Chemformer and its pretrained weights are Apache-2.0. **The GLMR repository has no
LICENSE file.** This port is the authors' own reimplementation, but the
redistribution terms of the GLMR code and of the trained GLMR checkpoints should be
stated explicitly before release.
