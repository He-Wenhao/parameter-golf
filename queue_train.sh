#!/bin/bash
# Sequential filter-training queue.
# Waits for current length run to finish, then runs the rest in order.
# Re-runnable: skips any with already-DONE logs and any with no data dir yet.

LOG=/workspace/queue.log
exec > "$LOG" 2>&1

wait_done() {
    local f="$1"
    while ! grep -q "=== DONE ===" "$f" 2>/dev/null; do
        if grep -qiE "traceback|runtimeerror|cuda error" "$f" 2>/dev/null; then
            echo "[queue] ERROR in $f"
            return 1
        fi
        sleep 20
    done
}

# Order: cheapest finished first. Skip ngram_rep (no-op).
for name in length entropy zlib_band stack_cheap alpha stopword stack_full; do
    log="/workspace/train_${name}.log"
    if grep -q "=== DONE ===" "$log" 2>/dev/null; then
        echo "[queue] $name already DONE, skip"
        continue
    fi
    if [[ ! -d "/workspace/data_${name}" ]] || [[ ! -f "/workspace/data_${name}/fineweb_train_000009.bin" ]]; then
        echo "[queue] $name not ready (data missing); deferring"
        continue
    fi
    if pgrep -f "RUN_ID=filter_${name}" >/dev/null; then
        echo "[queue] $name already running; waiting for it"
        wait_done "$log" || break
        continue
    fi
    echo "[queue] launching $name at $(date)"
    bash /workspace/run_filter_train.sh "$name"
    echo "[queue] $name done at $(date)"
done

echo "[queue] === FIRST-PASS DONE ==="
# Second pass for any deferred ones (slow filters that finished while we trained)
for name in alpha stopword stack_full; do
    log="/workspace/train_${name}.log"
    if grep -q "=== DONE ===" "$log" 2>/dev/null; then continue; fi
    if [[ ! -d "/workspace/data_${name}" ]] || [[ ! -f "/workspace/data_${name}/fineweb_train_000009.bin" ]]; then
        echo "[queue] $name STILL not ready, waiting up to 30 min"
        for _ in $(seq 1 90); do
            [[ -f "/workspace/data_${name}/fineweb_train_000009.bin" ]] && [[ -f "/workspace/data_${name}/filter_stats.json" ]] && break
            sleep 20
        done
    fi
    if [[ -f "/workspace/data_${name}/filter_stats.json" ]]; then
        echo "[queue] launching $name at $(date)"
        bash /workspace/run_filter_train.sh "$name"
        echo "[queue] $name done at $(date)"
    else
        echo "[queue] $name data still not ready; skip"
    fi
done
echo "[queue] === ALL DONE ==="
