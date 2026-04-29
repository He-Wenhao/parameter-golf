#!/bin/bash
# Queue v2: foreground sequential trainings only. No "already running" detection.
# Skip filters whose data isn't generated yet (slow filters handled by phase3).

LOG=/workspace/queue.log
exec > "$LOG" 2>&1

for name in length entropy zlib_band stack_cheap; do
    log="/workspace/train_${name}.log"
    if grep -q "=== DONE ===" "$log" 2>/dev/null; then
        echo "[queue] $name already DONE, skip"
        continue
    fi
    if [[ ! -d "/workspace/data_${name}" ]] || [[ ! -f "/workspace/data_${name}/fineweb_train_000009.bin" ]]; then
        echo "[queue] $name no data yet; skip"
        continue
    fi
    echo "[queue] launching $name at $(date)"
    bash /workspace/run_filter_train.sh "$name"
    echo "[queue] $name done at $(date)"
done

echo "[queue] === FIRST-PASS DONE ==="
echo "[queue] === ALL DONE ==="
