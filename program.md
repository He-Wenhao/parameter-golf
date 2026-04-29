# MDLM Diffusion for Parameter Golf

## Current State

**val_bpb: 1.8195** (4096-seq ELBO, int8+zlib) | 17.7M params | 9L 512d 8h GQA-2 | 5954 steps on 4xA100

**Problem: artifact is 16.52MB, cap is 16.00MB (decimal). Must shrink before submitting.**

Competitive landscape (diffusion PRs only, corrected BPB):
- PR #1403: 1.3485 — standard AR stack (Muon, U-Net, relu^2) adapted for bidirectional MDLM, 8xH100
- PR #1053: 1.3600 — pseudo-log-likelihood eval (8 masked passes), 8xH100
- PR #1119: 1.4584 — hybrid 50% MDLM + 50% NTP training, causal attention, sliding window AR eval, int6 GPTQ
- PR #1198: 1.5992 — hybrid sparse diffusion, 100K steps (2h)
- **Us: 1.8195** — pure bidirectional MDLM, uniform sigma importance sampling, AdamW, 6K steps on 4xA100

## How to Run

```bash
# Clone and setup
git clone https://github.com/He-Wenhao/parameter-golf.git
cd parameter-golf
git checkout autoresearch/auto-diffusion
python3 -m venv .venv && source .venv/bin/activate
pip install torch numpy sentencepiece huggingface-hub datasets tqdm

# Download data
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10

# Train on 8xH100 (default config)
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee run.log

# Train with custom config (all via env vars)
NUM_LAYERS=8 LR=1.1e-3 NOISE_EPS=0.01 BATCH_SIZE_PER_GPU=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee run.log
```

Key env vars: `NUM_LAYERS`, `MODEL_DIM`, `NUM_HEADS`, `NUM_KV_GROUPS`, `MLP_MULT`, `COND_DIM`, `LR`, `NOISE_EPS`, `BATCH_SIZE_PER_GPU`, `MAX_WALLCLOCK_SECONDS`, `ELBO_EVAL_STEPS`, `FINAL_EVAL_SEQS`, `SEED`.

Training takes 600s (wallclock cap), then runs final eval with 4096 sequences (~4 min). Total ~14 min.

Output: `final_model.pt`, `final_model.int8.ptz`, and logged metrics including `final_int8_zlib_roundtrip_exact`.

## Architecture

- Bidirectional transformer, RoPE, no causal mask
- SUBS parameterization: model predicts over real vocab only, frozen visible-token logits, zero mask probability
- AdaLN timestep conditioning: sigma -> MLP -> (scale, shift) per layer
- Grouped Query Attention (8 query heads, 2 KV groups)
- LeakyReLU(0.5)^2 activation
- Logit softcap=30, orthogonal init
- Per-row int8 quantization + zlib-9 compression

Training loss: uniform sigma importance sampling with antithetic pairs. Instead of sampling t~U[0,1] and reweighting by dsigma/dt (high variance), we sample sigma~U[0,sigma_max] with constant weight sigma_max.

Eval: discrete absorbing-mask ELBO with 128 Riemann steps.

## Improvement Priorities (ordered by expected impact)

### P0: Fix artifact size (BLOCKER)
Current: 16.52MB. Cap: 16.00MB. Must cut ~524KB.
- **Option A**: Drop to 8 layers (saves ~2M params = ~1.5MB compressed). Simple, proven (v4 was 8L at 15.9MB).
- **Option B**: Switch to int6 quantization + zstd (like PR #1119, fits 27M params in 11.5MB). Higher effort but allows much bigger models.
- **Option C**: Reduce model_dim from 512 to ~480, or MLP mult from 2.0 to 1.8.

### P1: More training steps (free win from 8xH100)
On 4xA100 we got 5954 steps. On 8xH100 expect ~12K steps at ~50ms/step. This alone should improve BPB by ~0.05-0.1. PR #1403 gets 11,808 steps on 8xH100.

### P2: Adopt Muon optimizer
All top competition entries (AR and diffusion) use Muon instead of AdamW. PR #1403 (best pure diffusion, 1.35) uses Muon. Muon enables ~10x higher LR and converges faster. This is probably the single biggest algorithmic improvement available.

Implementation: Muon is already in the competition codebase. Look at any top AR submission's train_gpt.py for the implementation. Key: Muon for weight matrices, AdamW for embeddings/norms/biases.

### P3: Adopt standard competition architecture tricks
PR #1403 (1.35 BPB) uses the full AR stack adapted for MDLM:
- U-Net skip connections between layers
- RMSNorm (instead of our AdaLN LayerNorm)
- relu^2 activation (we already have LeakyReLU(0.5)^2)
- Possibly 11 layers with int8+zlib fitting under 16MB

### P4: Int6 quantization + GPTQ
PR #1119 fits 27M params in 11.5MB using int6 GPTQ + zstd-22. This would let us run a much larger model (11L+, wider MLP) while staying under 16MB. GPTQ uses Hessian-aware quantization to minimize loss.

### P5: Alternative eval methods
Our 128-step ELBO is slow (~4 min) and gives a loose bound. Alternatives:
- **8-point trapezoidal quadrature** (PR #1403): much faster, slightly looser
- **Pseudo-log-likelihood** (PR #1053, 1.36 BPB): 8 forward passes with 50% masking. Not a true ELBO but gives better numbers.
- **Sliding window AR eval** (PR #1119, 1.46 BPB): requires causal attention. Use diffusion only as training regularizer, evaluate autoregressively. Best BPB but debatable whether it's "diffusion."

### P6: Hybrid training
PR #1119 trains 50% MDLM + 50% standard next-token prediction with causal attention. The diffusion acts as a regularizer. Eval is pure AR with sliding window. This gets 1.4584 BPB — much better than pure diffusion, but it's really an AR model with diffusion regularization.

### P7: Noise schedule tuning
- We use eps=0.01. PR #1403 uses eps=0.1. Our sweep showed eps=0.1 gave big gains at 500 steps but eps=0.01 was better at 6K steps (lower terminal KL). Re-sweep on 8xH100 with more steps.
- Try different noise schedules (cosine, linear vs log-linear).

### P8: EOS learning + shard rotation
PR #1241 adds EOS token learning (token 1 as document boundary, never masked) and trains on all 80 FineWeb shards via rotation. We currently only train on ~10 shards.

## Experiment Loop

Similar to autoresearch: modify train_gpt.py, run, check results, keep or discard.

```bash
# Run experiment
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee run.log

# Check results
grep "final_int8_zlib_roundtrip_exact" run.log
grep "Total submission size int8" run.log
grep "params:" run.log

# Key metrics to track:
# - val_bpb (lower is better)
# - artifact size (must be < 16,000,000 bytes)
# - steps completed (more = better utilization of 600s budget)
```

## Quick Reference: Competition Rules
- Artifact cap: 16,000,000 bytes (decimal, NOT 16 MiB)
- Wallclock: 600 seconds training on 8xH100
- Hardware: 8xH100 SXM
- Eval: tokenizer-agnostic BPB using sentencepiece 3-LUT byte counting
- Everything in one file: train_gpt.py
- No external downloads during eval
