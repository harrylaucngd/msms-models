#!/usr/bin/env bash
# Table 5, SMILES and SELFIES Transformer. Run from the repository root after `bash paper/prepare_data.sh`.
set -euo pipefail
here=paper/table5/smiles_transformer
molecules=data/MassSpecGym_molecules_MCES2_disjoint_with_test_fold_4M.tsv
for variant in smiles selfies; do
  out=${OUT:-results/paper/table5}/smiles_transformer_$variant
  tokenizer=$([ "$variant" = smiles ] && echo smiles-bpe || echo selfies)
  # The SMILES tokenizer is trained on 4M MassSpecGym molecules as well.
  [ "$variant" = selfies ] || [ -f "$molecules" ] || msms-models download msg --output data --molecules
  [ -f "checkpoints/smiles_transformer/$variant/model.pt" ] || \
    python -m msms_models.models.de_novo.smiles_transformer.train --tokenizer "$tokenizer" \
      --train data/msg/train.jsonl --validation data/msg/val.jsonl --molecules "$molecules" \
      --output "checkpoints/smiles_transformer/$variant" ${DEVICE:+--device "$DEVICE"}
  msms-models predict --config "$here/config-$variant.yaml" --output "$out" ${DEVICE:+--device "$DEVICE"}
  msms-models evaluate --protocol table5 --queries "$out/queries.jsonl" \
    --predictions "$out/predictions.jsonl" --output "$out/metrics.json" --mces-cache "$out/mces.sqlite"
done
