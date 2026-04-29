#!/bin/bash
# Train on a filtered dataset and log to per-filter log file.
# Usage: bash run_filter_train.sh <filter_name>
set -eo pipefail
NAME="${1:?need filter name}"
DATA_DIR="/workspace/data_$NAME"
[[ -d "$DATA_DIR" ]] || { echo "no data dir $DATA_DIR"; exit 1; }
LOG="/workspace/train_$NAME.log"
exec > "$LOG" 2>&1
echo "[$(date)] === TRAINING (filter=$NAME, 600s wallclock) ==="
echo "DATA_DIR=$DATA_DIR"
ls -la "$DATA_DIR"
cd /workspace/parameter-golf
RUN_ID="filter_${NAME}" \
  DATA_PATH="${DATA_DIR}/" \
  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
  VOCAB_SIZE=1024 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
echo "[$(date)] === DONE ==="
