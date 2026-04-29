#!/bin/bash
# Phase 3: regenerate slow CPU-bound filters (alpha/stopword/stack_full)
# AFTER phase2 finishes (no GPU competition / CPU competition).
# Then train each.

LOG=/workspace/phase3.log
exec > "$LOG" 2>&1
echo "[phase3] waiting for phase 2 ALL DONE..."
until grep -q "phase2.*ALL DONE ===" /workspace/phase2.log 2>/dev/null; do
    sleep 30
done
echo "[phase3] phase 2 done at $(date), regenerating slow filters in parallel"

cd /workspace/parameter-golf
SRC=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024
SP=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model
mkdir -p /workspace/filter_logs

# Wipe any partial data dirs
for name in alpha stopword stack_full; do
    rm -rf "/workspace/data_$name"
done

# Run all 3 in parallel — no training to compete with
for name in alpha stopword stack_full; do
    python3 filter_lib.py --name "$name" --src "$SRC" --dst "/workspace/data_$name" --sp "$SP" \
        > "/workspace/filter_logs/$name.log" 2>&1 &
done
wait
echo "[phase3] slow filters regenerated at $(date)"
for name in alpha stopword stack_full; do
    echo "--- $name ---"
    grep -A6 SUMMARY "/workspace/filter_logs/$name.log" | tail -10
done

# Train each sequentially
for name in alpha stopword stack_full; do
    if grep -q "=== DONE ===" "/workspace/train_${name}.log" 2>/dev/null; then
        echo "[phase3] $name already done, skip"; continue
    fi
    if [[ ! -f "/workspace/data_${name}/filter_stats.json" ]]; then
        echo "[phase3] no data for $name, skip"; continue
    fi
    echo "[phase3] training $name at $(date)"
    bash /workspace/run_filter_train.sh "$name"
    echo "[phase3] $name trained at $(date)"
done

echo "[phase3] === ALL DONE ==="
