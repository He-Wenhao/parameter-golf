#!/bin/bash
# Phase 4: re-run failed self-perplexity trainings after phase 3 finishes.
# Symlinks were broken (relative path) so those 3 trainings exited immediately.
LOG=/workspace/phase4.log
exec > "$LOG" 2>&1

echo "[phase4] fixing symlinks at $(date)"
VAL=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin
for name in selfppl_drop_high selfppl_drop_low selfppl_band; do
    target="/workspace/data_${name}/fineweb_val_000000.bin"
    rm -f "$target"
    ln -s "$VAL" "$target"
    ls -la "$target"
done

echo "[phase4] waiting for phase 3 ALL DONE..."
until grep -q "phase3.*ALL DONE ===" /workspace/phase3.log 2>/dev/null; do
    sleep 30
done
echo "[phase4] phase 3 done at $(date), running self-ppl trainings"

for name in selfppl_drop_high selfppl_drop_low selfppl_band; do
    log="/workspace/train_${name}.log"
    # Truncate the failed-training log so the wrapper rewrites it cleanly
    : > "$log"
    if [[ ! -f "/workspace/data_${name}/fineweb_train_000009.bin" ]]; then
        echo "[phase4] $name no data, skip"; continue
    fi
    echo "[phase4] training $name at $(date)"
    bash /workspace/run_filter_train.sh "$name"
    echo "[phase4] $name trained at $(date)"
done

echo "[phase4] === ALL DONE ==="
