"""
train_mdlm_combined.py — MDLM with competition-grade infrastructure.

Combines:
- PR #1053: Muon optimizer, weight tying, GQA, DDP, torch.compile, int8+zlib, always-decaying LR
- PR #1106: AdaLN timestep conditioning, log-linear noise schedule, SUBS parameterization, discrete ELBO eval
- Our contributions: uniform sigma importance sampling (eliminates dsigma variance), batch/step optimization

Usage:
  NUM_LAYERS=11 TRAIN_BATCH_TOKENS=131072 SEED=1337 torchrun --standalone --nproc_per_node=4 train_mdlm_combined.py
"""

from __future__ import annotations

import copy
import glob
import io
import math
import os
import sys
import time
import uuid
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP


# ============================================================
# HYPERPARAMETERS
# ============================================================

class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 20000))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))

    # Model architecture
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 11))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 2))
    cond_dim = int(os.environ.get("COND_DIM", 64))  # timestep conditioning dimension
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))

    # Optimizer (same as SOTA)
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))

    # MDLM diffusion hyperparameters
    noise_eps = float(os.environ.get("NOISE_EPS", 0.01))  # terminal noise floor
    elbo_eval_steps = int(os.environ.get("ELBO_EVAL_STEPS", 128))  # timesteps for ELBO eval
    max_eval_seqs = int(os.environ.get("MAX_EVAL_SEQS", 256))  # max sequences for eval
    use_dsigma_loss = bool(int(os.environ.get("USE_DSIGMA_LOSS", "0")))  # 0=importance sampling, 1=dsigma


# ============================================================
# MUON OPTIMIZER (same as train_gpt.py)
# ============================================================

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr, momentum, nesterov, backend_steps = (
                group["lr"], group["momentum"], group["nesterov"], group["backend_steps"],
            )
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if nesterov else buf
                if g.ndim == 2:
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                p.add_(g, alpha=-lr)


# ============================================================
# LOG-LINEAR NOISE SCHEDULE (from MDLM)
# ============================================================

def log_linear_noise(t: Tensor, eps: float) -> tuple[Tensor, Tensor]:
    """sigma(t) = -log(alpha(t)), alpha(t) = 1 - (1-eps)*t"""
    alpha = 1 - (1 - eps) * t
    sigma = -torch.log(alpha.clamp(min=1e-8))
    return sigma, alpha


# ============================================================
# TIMESTEP CONDITIONING (AdaLN from PR #1106)
# ============================================================

class TimestepEmbedder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))
        half = dim // 2
        self.register_buffer("freqs", torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half))

    def forward(self, sigma: Tensor) -> Tensor:
        emb = sigma[:, None] * self.freqs[None, :]
        return self.mlp(torch.cat([emb.sin(), emb.cos()], dim=-1))


class AdaLN(nn.Module):
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, 2 * dim, bias=True)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        s, sh = self.proj(c).unsqueeze(1).chunk(2, dim=-1)
        return F.rms_norm(x, (x.size(-1),)) * (1 + s) + sh


# ============================================================
# QUANTIZATION (same as train_gpt.py)
# ============================================================

INT8_CLIP_Q = 99.99984 / 100.0
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
CONTROL_TENSOR_NAME_PATTERNS = (
    "attn_scale", "attn_scales", "mlp_scale", "mlp_scales",
    "resid_mix", "resid_mixes", "q_gain", "skip_weight", "skip_weights",
    "adaln", "sigma_map",
)


def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel() else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = t32.clamp(-clip_abs[:, None], clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = clipped.div(scale[:, None]).round().clamp(-127, 127).to(torch.int8).contiguous()
        return q, scale.to(INT8_PER_ROW_SCALE_DTYPE).contiguous()
    clip_abs_val = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs_val / 127.0 if clip_abs_val > 0 else 1.0, dtype=torch.float32)
    q = t32.clamp(-clip_abs_val, clip_abs_val).div(scale).round().clamp(-127, 127).to(torch.int8).contiguous()
    return q, scale


def quantize_state_dict_int8(state_dict: dict[str, Tensor]) -> tuple[dict, dict]:
    quantized, scales, dtypes, passthrough, passthrough_orig_dtypes, qmeta = {}, {}, {}, {}, {}, {}
    stats = dict.fromkeys(("param_count", "num_tensors", "baseline_tensor_bytes", "int8_payload_bytes"), 0)
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        stats["param_count"] += t.numel()
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += t.numel() * t.element_size()
        if not t.is_floating_point():
            passthrough[name] = t
            stats["int8_payload_bytes"] += t.numel() * t.element_size()
            continue
        is_control = any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS)
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL or is_control:
            if t.dtype in {torch.float32, torch.bfloat16}:
                passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
                passthrough[name] = t.to(INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
            else:
                passthrough[name] = t
            stats["int8_payload_bytes"] += passthrough[name].numel() * passthrough[name].element_size()
            continue
        q, s = quantize_float_tensor(t)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += q.numel() + s.numel() * s.element_size()
    obj: dict = {"__quant_format__": "int8_clean_per_row_v1", "quantized": quantized,
                 "scales": scales, "dtypes": dtypes, "passthrough": passthrough}
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats


def dequantize_state_dict_int8(obj: dict) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            out[name] = (q.float() * s.to(torch.float32).view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype).contiguous()
        else:
            out[name] = (q.float() * float(s.item())).to(dtype).contiguous()
    for name, t in obj["passthrough"].items():
        out_t = t.detach().cpu().contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


# ============================================================
# DATA LOADING (same as train_gpt.py)
# ============================================================

def load_data_shard(file: Path) -> Tensor:
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header: {file}")
    num_tokens = int(header[2])
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=256 * 4)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No data files: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance(self):
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance()
                continue
            take = min(avail, remaining)
            chunks.append(self.tokens[self.pos: self.pos + take])
            self.pos += take
            remaining -= take
        return torch.cat(chunks) if len(chunks) > 1 else chunks[0]


class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.stream = TokenStream(pattern)
        self.device = device
        self.world_size = world_size

    def next_batch(self, total_tokens: int, seq_len: int, grad_accum_steps: int) -> Tensor:
        tokens_per_step = total_tokens // self.world_size // grad_accum_steps
        raw = self.stream.take(tokens_per_step)
        x0 = raw.view(-1, seq_len).long()
        return x0.to(self.device, non_blocking=True)


def load_validation_tokens(val_files: str, seq_len: int) -> Tensor:
    chunks = [load_data_shard(Path(p)) for p in sorted(glob.glob(val_files))]
    if not chunks:
        raise FileNotFoundError(f"No val files: {val_files}")
    tokens = torch.cat(chunks)
    n = (tokens.numel() // seq_len) * seq_len
    return tokens[:n]


# ============================================================
# MODEL (same transformer backbone as train_gpt.py)
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        self.register_buffer("inv_freq", (1.0 / base ** (torch.arange(0, dim, 2).float() / dim)).float())

    def forward(self, seqlen: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: Tensor) -> Tensor:
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    return x * cos + rotate_half(x) * sin


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, rope_base: float, qk_gain_init: float):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x: Tensor, is_causal: bool = True) -> Tensor:
        B, T, D = x.shape
        q = self.c_q(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(T, x.device, q.dtype)
        q = apply_rotary(q, cos, sin) * self.q_gain.to(q.dtype)[None, :, None, None]
        k = apply_rotary(k, cos, sin)
        groups = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        return self.proj(y.transpose(1, 2).reshape(B, T, D))


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        self.fc = CastedLinear(dim, mlp_mult * dim, bias=False)
        self.proj = CastedLinear(mlp_mult * dim, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(torch.relu(self.fc(x)).square())


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int,
                 rope_base: float, qk_gain_init: float, cond_dim: int):
        super().__init__()
        self.adaln_attn = AdaLN(dim, cond_dim)
        self.attn = Attention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.adaln_mlp = AdaLN(dim, cond_dim)
        self.mlp = MLP(dim, mlp_mult)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        x = x + self.attn(self.adaln_attn(x, c), is_causal=False)
        x = x + self.mlp(self.adaln_mlp(x, c))
        return x


class MDLM(nn.Module):
    """
    Masked Diffusion Language Model with AdaLN timestep conditioning.
    Combines PR #1053 architecture (Muon, GQA, weight tying) with
    PR #1106 MDLM training (noise schedule, SUBS parameterization, ELBO eval).
    """

    def __init__(
        self,
        vocab_size: int,
        model_dim: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        cond_dim: int,
        rope_base: float,
        logit_softcap: float,
        qk_gain_init: float,
        tie_embeddings: bool,
        tied_embed_init_std: float,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.mask_id = vocab_size
        self.logit_softcap = logit_softcap
        self.tie_embeddings = tie_embeddings

        self.tok_emb = nn.Embedding(vocab_size + 1, model_dim)
        self.sigma_map = TimestepEmbedder(cond_dim)
        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init, cond_dim)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm()

        if tie_embeddings:
            self.lm_head = None
        else:
            self.lm_head = CastedLinear(model_dim, vocab_size, bias=False)
            self.lm_head._zero_init = True

        self._init_weights(tied_embed_init_std)

    def _init_weights(self, tied_embed_init_std: float):
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=tied_embed_init_std)
        for m in self.modules():
            if isinstance(m, CastedLinear):
                if getattr(m, "_zero_init", False):
                    nn.init.zeros_(m.weight)
                else:
                    nn.init.normal_(m.weight, std=0.02)

    def forward(self, xt: Tensor, sigma: Tensor) -> Tensor:
        """Forward pass returning SUBS log probs. xt may contain MASK tokens."""
        x = self.tok_emb(xt)
        c = F.silu(self.sigma_map(sigma)).to(dtype=x.dtype)
        for block in self.blocks:
            x = block(x, c)
        x = self.final_norm(x)

        if self.tie_embeddings:
            logits = x @ self.tok_emb.weight[:self.vocab_size].to(x.dtype).T
        else:
            logits = self.lm_head(x)

        if self.logit_softcap > 0:
            logits = torch.tanh(logits / self.logit_softcap) * self.logit_softcap

        # SUBS parameterization: frozen visible tokens, zero mask probability
        logits = logits.float()
        log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        frozen = torch.full_like(log_probs, -1e6)
        # Clamp scatter index: MASK positions (id=vocab_size) are out of bounds for logits,
        # but their frozen values are unused (overwritten by log_probs via visible mask)
        frozen.scatter_(-1, xt.clamp(max=self.vocab_size - 1)[..., None], 0.0)
        visible = (xt != self.mask_id)[..., None]
        return torch.where(visible, frozen, log_probs)


# ============================================================
# MDLM LOSS (uniform sigma importance sampling)
# ============================================================

def mdlm_loss(model: nn.Module, x0: Tensor, noise_eps: float, mask_id: int) -> Tensor:
    """MDLM continuous-time ELBO loss with uniform sigma importance sampling.

    Instead of sampling t ~ U[0,1] and weighting by dsigma/dt (which has high variance),
    we sample sigma ~ U[0, sigma_max] directly. This gives a constant weight of sigma_max,
    eliminating the dsigma variance that plagues standard MDLM training.

    model can be a raw MDLM, compiled, or DDP-wrapped model.
    """
    B, L = x0.shape
    sigma_max = -math.log(noise_eps)
    # Antithetic sampling in sigma space for variance reduction
    sigma = torch.rand(B // 2 + 1, device=x0.device) * sigma_max
    sigma = torch.cat([sigma, sigma_max - sigma])[:B]
    alpha = torch.exp(-sigma)
    # Mask tokens with probability 1 - alpha(sigma)
    xt = torch.where(torch.rand(B, L, device=x0.device) < (1 - alpha[:, None]), mask_id, x0)
    log_probs = model(xt, sigma)
    log_p_x0 = torch.gather(log_probs, -1, x0[..., None]).squeeze(-1)
    is_masked = (xt == mask_id).float()
    # sigma_max factor from importance sampling: E_sigma[f(sigma)] = sigma_max * E_U[f]
    return sigma_max * ((-log_p_x0) * is_masked).sum() / (B * L)


def mdlm_loss_dsigma(model: nn.Module, x0: Tensor, noise_eps: float, mask_id: int) -> Tensor:
    """Standard MDLM loss with dsigma/dt weighting (as in Sahoo et al., PR #1106)."""
    B, L = x0.shape
    # Antithetic sampling in t space
    t = torch.rand(B // 2 + 1, device=x0.device)
    t = torch.cat([t, 1 - t])[:B].clamp(1e-5, 1 - 1e-5)
    sigma, alpha = log_linear_noise(t, noise_eps)
    xt = torch.where(torch.rand(B, L, device=x0.device) < (1 - alpha[:, None]), mask_id, x0)
    log_probs = model(xt, sigma)
    log_p_x0 = torch.gather(log_probs, -1, x0[..., None]).squeeze(-1)
    dsigma = (1 - noise_eps) / alpha
    is_masked = (xt == mask_id).float()
    return (dsigma[:, None] * (-log_p_x0) * is_masked).sum() / (B * L)


# ============================================================
# DISCRETE ELBO EVALUATION
# ============================================================

@torch.no_grad()
def discrete_elbo_eval(
    model: MDLM,
    val_tokens: Tensor,
    seq_len: int,
    noise_eps: float,
    n_steps: int,
    max_seqs: int,
    base_bytes_lut: Tensor,
    device: torch.device,
    rank: int,
    world_size: int,
) -> tuple[float, float]:
    """Discrete absorbing-mask variational ELBO (from MDLM / PR #820).

    Discretizes the continuous-time ELBO into n_steps timesteps.
    More steps = tighter bound. Returns (val_loss_nats_per_token, val_bpb).
    """
    model.eval()
    n_seq = min(val_tokens.numel() // seq_len, max_seqs)

    seq_per_rank = (n_seq + world_size - 1) // world_size
    start = rank * seq_per_rank
    end = min(start + seq_per_rank, n_seq)

    total_bits = torch.zeros((), device=device, dtype=torch.float64)
    total_bytes = torch.zeros((), device=device, dtype=torch.float64)
    total_tokens = torch.zeros((), device=device, dtype=torch.float64)

    t_grid = torch.arange(1, n_steps + 1, device=device, dtype=torch.float32) / n_steps
    sigma_grid, alpha_grid = log_linear_noise(t_grid, noise_eps)

    vocab_size = model.vocab_size
    mask_id = model.mask_id

    for i in range(start, end):
        x0 = val_tokens[i * seq_len:(i + 1) * seq_len].long().unsqueeze(0).to(device)
        seq_bits = torch.zeros(1, device=device, dtype=torch.float64)

        # Terminal KL: remaining probability at t=1
        alpha_T = float(alpha_grid[-1])
        seq_bits += seq_len * alpha_T * math.log(vocab_size) / math.log(2.0)

        alpha_prev = 1.0
        for step in range(n_steps):
            alpha_curr = float(alpha_grid[step])
            sigma_curr = sigma_grid[step].unsqueeze(0)
            move_chance = 1 - alpha_curr

            xt = torch.where(torch.rand_like(x0.float()) < move_chance, mask_id, x0)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                log_probs = model(xt, sigma_curr)
            log_p_x0 = torch.gather(log_probs.float(), -1, x0[..., None]).squeeze(-1)

            reveal_prob = (alpha_prev - alpha_curr) / max(1.0 - alpha_curr, 1e-12)
            is_masked = (xt == mask_id).float()
            step_bits = reveal_prob * (-log_p_x0) * is_masked / math.log(2.0)
            seq_bits += step_bits.sum(dim=-1).to(torch.float64)

            alpha_prev = alpha_curr

        total_bits += seq_bits.sum()
        total_bytes += base_bytes_lut[x0.view(-1).cpu()].to(torch.float64).sum().to(device)
        total_tokens += seq_len

    if world_size > 1:
        dist.all_reduce(total_bits, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)

    val_bpb = float(total_bits.item() / max(total_bytes.item(), 1))
    val_loss = float(total_bits.item() * math.log(2.0) / max(total_tokens.item(), 1))
    model.train()
    return val_loss, val_bpb


# ============================================================
# MAIN
# ============================================================

def main():
    distributed = int(os.environ.get("RANK", -1)) != -1
    if distributed:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    master_process = rank == 0
    device = torch.device("cuda", local_rank)

    def log0(msg: str):
        if master_process:
            print(msg, flush=True)

    args = Hyperparameters()
    torch.manual_seed(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    # Tokenizer for BPB
    sp = spm.SentencePieceProcessor()
    sp.Load(args.tokenizer_path)
    base_bytes_lut = torch.zeros(args.vocab_size, dtype=torch.int32)
    for i in range(args.vocab_size):
        base_bytes_lut[i] = len(sp.IdToPiece(i).replace("\u2581", " ").encode("utf-8"))
    base_bytes_lut = base_bytes_lut.to(device)

    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len).to(device)
    log0(f"val_tokens:{val_tokens.numel()} val_sequences:{val_tokens.numel()//args.train_seq_len}")

    total_batch = args.train_batch_tokens
    seqs_per_step = total_batch // world_size // args.train_seq_len
    grad_accum_steps = max(1, seqs_per_step // 32)
    micro_seqs = seqs_per_step // grad_accum_steps
    log0(f"grad_accum:{grad_accum_steps} micro_seqs:{micro_seqs}")

    base_model = MDLM(
        vocab_size=args.vocab_size,
        model_dim=args.model_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        cond_dim=args.cond_dim,
        rope_base=args.rope_base,
        logit_softcap=args.logit_softcap,
        qk_gain_init=args.qk_gain_init,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
    ).to(device).bfloat16()

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"MDLM: {args.num_layers}L {args.model_dim}d {args.num_heads}h GQA-{args.num_kv_heads} "
         f"cond_dim:{args.cond_dim} — {n_params:,} params")

    for m in base_model.modules():
        if isinstance(m, CastedLinear):
            m.float()
        if isinstance(m, Rotary):
            m.inv_freq.data = m.inv_freq.data.float()
    # q_gain, norms, and small tensors stay fp32
    for name, p in base_model.named_parameters():
        if p.ndim < 2 and p.dtype != torch.float32:
            p.data = p.data.float()

    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model: nn.Module = (
        DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False)
        if distributed else compiled_model
    )

    # Optimizer param groups
    # Muon: 2D block params (attention weights, MLP weights, AdaLN projection weights)
    matrix_params = [p for p in base_model.blocks.parameters() if p.ndim == 2]
    muon_ids = {id(p) for p in matrix_params}

    # Embedding params
    if args.tie_embeddings:
        embed_params = [base_model.tok_emb.weight]
        token_lr = args.tied_embed_lr
    else:
        embed_params = [base_model.tok_emb.weight]
        extra = list(base_model.lm_head.parameters()) if base_model.lm_head else []
        embed_params.extend(extra)
        token_lr = args.embed_lr
    embed_ids = {id(p) for p in embed_params}

    # Everything else → Adam scalar_lr (block 1D params like q_gain/adaln biases,
    # sigma_map params, final_norm — all non-Muon, non-embedding params)
    scalar_params = [p for p in base_model.parameters()
                     if id(p) not in muon_ids and id(p) not in embed_ids]

    optimizer_muon = Muon(
        [{"params": matrix_params, "base_lr": args.matrix_lr}],
        lr=args.matrix_lr, momentum=args.muon_momentum, backend_steps=args.muon_backend_steps,
    )
    optimizer_adam = torch.optim.Adam(
        [
            {"params": embed_params, "base_lr": token_lr, "lr": token_lr},
            {"params": scalar_params, "base_lr": args.scalar_lr, "lr": args.scalar_lr},
        ],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = group["lr"]
    optimizers = [optimizer_muon, optimizer_adam]

    # Verify all params have an optimizer
    optimized_ids = set()
    for opt in optimizers:
        for group in opt.param_groups:
            for p in group["params"]:
                optimized_ids.add(id(p))
    all_ids = {id(p) for p in base_model.parameters()}
    assert optimized_ids == all_ids, f"Missing params: {len(all_ids - optimized_ids)} unoptimized"

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all():
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            ws = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) \
                if ws <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    loss_fn = mdlm_loss_dsigma if args.use_dsigma_loss else mdlm_loss
    log0(f"MDLM training: noise_eps={args.noise_eps} loss={'dsigma' if args.use_dsigma_loss else 'importance'} "
         f"elbo_eval_steps={args.elbo_eval_steps} max_eval_seqs={args.max_eval_seqs}")
    log0(f"vocab_size:{args.vocab_size}+1(MASK) num_layers:{args.num_layers} "
         f"model_dim:{args.model_dim} tie_embeddings:{args.tie_embeddings}")
    log0(f"Muon matrix params: {sum(p.numel() for p in matrix_params):,} | "
         f"Adam embed params: {sum(p.numel() for p in embed_params):,} | "
         f"Adam scalar params: {sum(p.numel() for p in scalar_params):,}")

    # Warmup (run a few steps then reset to warm up torch.compile and optimizers)
    if args.warmup_steps > 0:
        init_state = {n: t.detach().cpu().clone() for n, t in base_model.state_dict().items()}
        init_opts = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for _ in range(args.warmup_steps):
            zero_grad_all()
            x0 = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = loss_fn(base_model, x0, args.noise_eps, base_model.mask_id)
            loss.backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
        base_model.load_state_dict(init_state, strict=True)
        for opt, state in zip(optimizers, init_opts):
            opt.load_state_dict(state)
        zero_grad_all()
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # Training loop
    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = 0

    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = discrete_elbo_eval(
                base_model, val_tokens, args.train_seq_len, args.noise_eps,
                args.elbo_eval_steps, args.max_eval_seqs, base_bytes_lut,
                device, rank, world_size,
            )
            log0(f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                 f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step,1):.2f}ms")
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        zero_grad_all()

        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x0 = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = loss_fn(model, x0, args.noise_eps, base_model.mask_id) / grad_accum_steps
            loss.backward()
            train_loss += loss.detach()

        frac = min(step / max(args.muon_momentum_warmup_steps, 1), 1.0)
        muon_mom = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_mom
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        step += 1
        approx_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0):
            log0(f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                 f"train_time:{approx_ms:.0f}ms step_avg:{approx_ms/step:.2f}ms")

        reached_cap = max_wallclock_ms is not None and approx_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            cap_t = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(cap_t, op=dist.ReduceOp.MAX)
            reached_cap = bool(cap_t.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    # Export int8+zlib
    if master_process:
        torch.save(base_model.state_dict(), "final_diffusion.pt")
    quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict())
    buf = io.BytesIO()
    torch.save(quant_obj, buf)
    blob = zlib.compress(buf.getvalue(), level=9)
    if master_process:
        with open("final_diffusion.int8.ptz", "wb") as f:
            f.write(blob)
        code_bytes = len(open(__file__).read().encode("utf-8"))
        log0(f"artifact:{len(blob)} bytes ({len(blob)/1e6:.1f}MB) code:{code_bytes} "
             f"total:{len(blob)+code_bytes} bytes")

    if distributed:
        dist.barrier()

    # Roundtrip validation: load quantized model and re-eval
    quant_state = torch.load(io.BytesIO(zlib.decompress(blob)), map_location="cpu")
    base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)
    val_loss_q, val_bpb_q = discrete_elbo_eval(
        base_model, val_tokens, args.train_seq_len, args.noise_eps,
        args.elbo_eval_steps, args.max_eval_seqs, base_bytes_lut,
        device, rank, world_size,
    )
    log0(f"final_int8_roundtrip val_loss:{val_loss_q:.4f} val_bpb:{val_bpb_q:.4f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
