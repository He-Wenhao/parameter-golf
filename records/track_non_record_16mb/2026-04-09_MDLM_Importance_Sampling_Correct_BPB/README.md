# Non-Record: MDLM with Uniform Sigma Importance Sampling

**val_bpb: 1.7657** (int8+zlib, 128 ELBO steps) | **19.0M params** | 4xA100-SXM4-40GB | Non-record

## Key Innovation: Uniform Sigma Importance Sampling

Standard MDLM training samples timestep t ~ U[0,1] and reweights the loss by dsigma/dt = (1-eps)/alpha(t). This produces high-variance gradients because alpha(t) -> eps near t=1.

We reformulate the ELBO integral via a change of variable: instead of integrating over t, we integrate over sigma directly. Sampling sigma ~ U[0, sigma_max] gives a constant weight of sigma_max, completely eliminating the dsigma variance:

```
Standard:  L = E_t [ dsigma/dt * loss(t) ]          # variance ~ 1/alpha(t)^2
Ours:      L = sigma_max * E_sigma [ loss(sigma) ]   # constant weight
```

Combined with antithetic sampling (pairing sigma with sigma_max - sigma), this gives significantly more stable training. In our sweep, importance sampling outperformed dsigma weighting by 0.30 BPB at 500 steps.

## Results

| Metric | Value |
|--------|-------|
| val_bpb (int8+zlib) | **1.7657** |
| val_bpb (pre-quant) | 1.7621 |
| val_loss (nats/tok) | 2.9812 |
| Steps | 6,372 (600s wallclock) |
| Artifact | 15.96 MB |
| Parameters | 18,989,376 |

Note on BPB: we use the same tokenizer-agnostic 3-LUT byte counting as the AR baseline (`build_sentencepiece_luts`). Prior diffusion work (PR #1106) used a hardcoded 4.3 bytes/token approximation; the actual weighted average for SP1024 is 2.46 bytes/token.

## Architecture

- 8 layers, 512 dim, 8 heads (GQA-2), 2x MLP, LeakyReLU(0.5)^2
- AdaLN timestep conditioning (sigma -> scale/shift per layer)
- Bidirectional attention with RoPE, no causal mask
- SUBS parameterization with frozen visible-token logits
- Orthogonal init, logit softcap=30

## Training

- Uniform sigma importance sampling + antithetic pairs
- AdamW (lr=1.1e-3, betas=0.9/0.95, wd=0.1), grad clip=1.0
- Wallclock-based cosine warmdown
- 262K tokens/step (64 seqs x 4 GPUs x 1024 tokens)
- Per-row int8 quantization + zlib-9 (0.004 BPB degradation)

## Sweep Summary (27 experiments)

| Change | BPB delta |
|--------|-----------|
| noise_eps 0.01->0.1 | **-0.32** |
| Importance sampling vs dsigma | **-0.30** |
| 9L 3x MLP (wider) | -0.35 (but >16MB) |
| Logit softcap on | -0.44 |
| seq_len 1024->2048 | -0.05 |
| cond_dim 64->128 | +0.91 |

Best config that fits 16MB: 8L 512d GQA-2, eps=0.01, importance sampling, softcap=30.

## The Diffusion-AR Gap

Diffusion (1.77) vs AR (1.22) — a 0.55 BPB gap. This is expected: the discrete ELBO is an upper bound on NLL, while AR computes exact NLL. With 19M params and 6K training steps, the model cannot fully learn all noise levels. Closing this gap likely requires larger models, longer training, or tighter variational bounds.

## Credits

- Sahoo et al. (2024), "Simple and Effective Masked Diffusion Language Models"
- PR #820: first MDLM in parameter-golf, discrete ELBO eval
- PR #1106: MDLM with adaLN conditioning
- PR #1053: int8+zlib quantization, distributed training infrastructure
