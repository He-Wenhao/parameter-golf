#!/bin/bash
# Phase 2: self-perplexity scoring + filtering + training.
# Waits for queue.log to contain ALL DONE, then runs scoring and 3 ppl-based runs.

LOG=/workspace/phase2.log
exec > "$LOG" 2>&1
echo "[phase2] waiting for phase 1 ALL DONE..."
until grep -q "ALL DONE ===" /workspace/queue.log 2>/dev/null; do
    sleep 30
done
echo "[phase2] phase 1 done at $(date), starting scoring"

cd /workspace/parameter-golf

# 1) Score all shards (uses 1 GPU; ~5 min)
python3 score_self_ppl.py --src ./data/datasets/fineweb10B_sp1024 --out /workspace/scores 2>&1 | tee /workspace/score.log

# 2) Build self-ppl filtered datasets
mkdir -p /workspace/sppl_logs
for mode in drop_high drop_low band; do
    case $mode in
        drop_high|drop_low) args="--q 0.10";;
        band) args="--lo 0.10 --hi 0.90";;
    esac
    name="selfppl_${mode}"
    python3 filter_by_scores.py --src ./data/datasets/fineweb10B_sp1024 --scores /workspace/scores \
        --dst "/workspace/data_${name}" --mode "$mode" $args > "/workspace/sppl_logs/${name}.log" 2>&1
    echo "[phase2] filter $name built"
    grep -A6 SUMMARY "/workspace/sppl_logs/${name}.log" | tail -10
done

# 3) Train each in sequence
for mode in drop_high drop_low band; do
    name="selfppl_${mode}"
    if grep -q "=== DONE ===" "/workspace/train_${name}.log" 2>/dev/null; then
        echo "[phase2] $name already done, skip"; continue
    fi
    if [[ ! -d "/workspace/data_${name}" ]] || [[ ! -f "/workspace/data_${name}/fineweb_train_000009.bin" ]]; then
        echo "[phase2] data missing for $name, skip"; continue
    fi
    echo "[phase2] training $name at $(date)"
    bash /workspace/run_filter_train.sh "$name"
    echo "[phase2] $name trained at $(date)"
done

echo "[phase2] === ALL DONE ==="
