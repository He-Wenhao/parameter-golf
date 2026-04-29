#!/bin/bash
# Generate all 8 heuristic filtered datasets in parallel.
set -eo pipefail
cd /workspace/parameter-golf
SRC=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024
SP=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model
mkdir -p /workspace/filter_logs
for name in length ngram_rep entropy zlib_band alpha stopword stack_cheap stack_full; do
    python3 filter_lib.py --name "$name" --src "$SRC" --dst "/workspace/data_$name" --sp "$SP" \
        > "/workspace/filter_logs/$name.log" 2>&1 &
done
wait
echo "=== ALL FILTERS DONE ==="
for name in length ngram_rep entropy zlib_band alpha stopword stack_cheap stack_full; do
    echo "--- $name ---"
    grep -A20 SUMMARY "/workspace/filter_logs/$name.log" | head -15
done
