# autoresearch — parameter-golf

This is an experiment to have the LLM do its own research on the parameter-golf challenge.

**Goal**: Train the best language model that fits in a **16MB artifact** and trains in under **10 minutes on 4×A100 GPUs**, evaluated by **compression on FineWeb validation set** (bits-per-byte, val_bpb). Lower is better.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr7`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current HEAD.
3. **Read the in-scope files**: Read these files for full context:
   - `README.md` — challenge rules, leaderboard, and context.
   - `train_gpt.py` — the file you modify. Model architecture, optimizer, training loop, evaluation, quantization. (~1500 lines)
4. **Verify data exists**: Check that `./data/datasets/fineweb10B_sp1024/` contains training shards and `./data/tokenizers/fineweb_1024_bpe.model` exists. If not, tell the human to run `python data/cached_challenge_fineweb.py --variant sp1024`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on **4 GPUs** via torchrun. The training script runs for a **fixed time budget of 10 minutes** (wall clock training time, enforced by MAX_WALLCLOCK_SECONDS=600).

**Important: HPC environment.** You are on a NERSC Perlmutter login node and CANNOT run GPU jobs directly. You must submit jobs to a compute node using `srun`. Launch experiments as:

```
srun --nodes 1 --qos interactive --time 00:30:00 --constraint gpu --gpus-per-node 4 --account m3706_g \
  .venv/bin/torchrun --standalone --nproc_per_node=4 train_gpt.py > run.log 2>&1
```

You can pass configuration via environment variables (see below), e.g.:
```
RUN_ID=experiment_name NUM_LAYERS=12 MODEL_DIM=384 srun ... \
  .venv/bin/torchrun --standalone --nproc_per_node=4 train_gpt.py > run.log 2>&1
```

**Important**: Use `.venv/bin/torchrun` (the project's virtualenv) — do NOT use `module load pytorch` as it has a different Python with missing dependencies.

**Key environment variables** (all optional, have sensible defaults):
- `RUN_ID` — experiment name for logs
- `NUM_LAYERS` (default 9), `MODEL_DIM` (default 512), `NUM_HEADS` (default 8), `NUM_KV_HEADS` (default 4)
- `MLP_MULT` (default 2), `TIE_EMBEDDINGS` (default 1)
- `MATRIX_LR` (default 0.04), `SCALAR_LR` (default 0.04), `EMBED_LR` (default 0.6)
- `WARMUP_STEPS` (default 20), `WARMDOWN_ITERS` (default 1200)
- `MUON_MOMENTUM` (default 0.95), `MUON_BACKEND_STEPS` (default 5)
- `VOCAB_SIZE` (default 1024), `TRAIN_SEQ_LEN` (default 1024)
- `MAX_WALLCLOCK_SECONDS` (default 600.0)
- `ITERATIONS` (default 20000)

**What you CAN do:**
- Modify `train_gpt.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, quantization, etc.
- Change environment variables to tune hyperparameters without code changes.

**What you CANNOT do:**
- Modify the data preparation or tokenizer. The validation set and tokenizer are fixed.
- Install new packages. Only what's in `requirements.txt` is available.
- Exceed the 16MB artifact size limit (code + `final_model.int8.ptz`).

**The goal is simple: get the lowest val_bpb.** Since the time budget is fixed at 10 minutes, you don't need to worry about training time. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size, the quantization. The constraints are: the code runs without crashing, finishes within the time budget, and the artifact fits in 16MB.

**Artifact size constraint**: After training, the script exports `final_model.int8.ptz` (int8 quantized + zlib compressed). The total of `train_gpt.py` (UTF-8 bytes) + `final_model.int8.ptz` (file size) must be ≤ 16,000,000 bytes.

**VRAM** is a soft constraint. The compute nodes have 4×A100-40GB. Some VRAM increase is acceptable for meaningful val_bpb gains, but OOM crashes are failures.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

The training script logs progress like:

```
step:1000/20000 val_loss:2.3345 val_bpb:1.3826 train_time:174066ms step_avg:174.07ms
```

At the end it performs int8 quantization + zlib compression and evaluates the roundtrip:

```
final_int8_zlib_roundtrip val_loss:2.1394 val_bpb:1.2671 eval_time:5929ms
final_int8_zlib_roundtrip_exact val_loss:2.13936859 val_bpb:1.26705458
```

**The `final_int8_zlib_roundtrip_exact val_bpb` is the ground truth metric.** This is what matters for the leaderboard.

You can extract the key metrics from the log file:

```
grep "final_int8_zlib_roundtrip_exact" run.log
```

And check artifact size:
```
ls -la final_model.int8.ptz | awk '{print $5}'
wc -c < train_gpt.py
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 6 columns:

```
commit	val_bpb	memory_gb	artifact_mb	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved (from `final_int8_zlib_roundtrip_exact`) — use 0.000000 for crashes
3. peak memory in GB, round to .1f — use 0.0 for crashes
4. artifact size in MB, round to .1f (train_gpt.py bytes + final_model.int8.ptz bytes, divided by 1e6) — use 0.0 for crashes
5. status: `keep`, `discard`, or `crash`
6. short text description of what this experiment tried

Example:

```
commit	val_bpb	memory_gb	artifact_mb	status	description
a1b2c3d	1.267054	44.0	15.8	keep	baseline
b2c3d4e	1.253200	44.2	15.9	keep	increase LR to 0.06
c3d4e5f	1.275000	44.0	15.8	discard	switch to GeLU activation
d4e5f6g	0.000000	0.0	0.0	crash	double model width (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/apr7`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train_gpt.py` with an experimental idea by directly hacking the code (or adjust env vars).
3. git commit
4. Run the experiment:
   ```
   srun --nodes 1 --qos interactive --time 00:30:00 --constraint gpu --gpus-per-node 4 --account m3706_g \
     .venv/bin/torchrun --standalone --nproc_per_node=4 train_gpt.py > run.log 2>&1
   ```
   (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "final_int8_zlib_roundtrip_exact\|peak_vram_mb" run.log` (if available) or check the last val_bpb logged.
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up on that idea.
7. Also check artifact size: `ls -la final_model.int8.ptz | awk '{print $5}'` — if it exceeds ~15.95MB (leaving room for code), the experiment violates constraints.
8. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
9. If val_bpb improved (lower), you "advance" the branch, keeping the git commit
10. If val_bpb is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~10 minutes of training + a few minutes for startup/eval overhead. If a run exceeds 20 minutes total, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read train_gpt.py for new angles, study the records/ directory for techniques used by top submissions, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

## Research strategy

When looking for ideas, consider:
- Study the `records/` directory — top submissions contain READMEs explaining their techniques
- Architecture changes: depth, width, attention variants, MLP variants, skip connections, parameter tying/sharing
- Optimizer tuning: learning rates, warmup/warmdown schedules, momentum
- Quantization-aware approaches: techniques that compress better (GPTQ, mixed precision, etc.)
- Embedding tricks: vocabulary handling, weight tying strategies
- Training tricks: batch size, sequence length, gradient accumulation
- The current SOTA is ~1.1147 val_bpb — there's a lot of room to improve from the baseline (~1.267)
