#!/usr/bin/env bash
# Figure 2, decoder querying with MS-GPT. Run from the repository root after `bash paper/prepare_data.sh`.
set -euo pipefail
here=paper/figure2/ms_gpt
# The released MS-GPT decoder and SAFE tokenizer, from the author-authorized mirrors.
hf download nielsr/MS-GPT-MLM --revision a139951c6cd2b6f31be9a53623a869610ee1d7ec \
  --local-dir checkpoints/ms-gpt/decoder
hf download nielsr/MS-GPT-SAFE-GPT-Base tokenizer.json \
  --revision 8d0560e93964a3344e724153c1bc83178d4a9d8f --local-dir checkpoints/ms-gpt
out=${OUT:-results/paper/figure2/ms_gpt}
msms-models predict --config "$here/config.yaml" --output "$out" ${DEVICE:+--device "$DEVICE"}
msms-models evaluate --protocol table5 --queries "$out/queries.jsonl" \
  --predictions "$out/predictions.jsonl" --output "$out/metrics.json" --mces-cache "$out/mces.sqlite"
