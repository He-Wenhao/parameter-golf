"""
MDLM (Masked Diffusion Language Model) for Parameter Golf.
Bidirectional transformer with log-linear noise schedule, adaLN timestep
conditioning, and discrete absorbing-mask ELBO evaluation.
Based on Sahoo et al., "Simple and Effective Masked Diffusion Language Models" (NeurIPS 2024).
"""

from __future__ import annotations

import copy
import glob
import io
import math
import os
import random
import subprocess
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

# -----------------------------
# HYPERPARAMETERS
# -----------------------------

class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 500))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 100))

    # Training length.
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 200))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    batch_size_per_gpu = int(os.environ.get("BATCH_SIZE_PER_GPU", 64))
    grad_accum_steps = int(os.environ.get("GRAD_ACCUM_STEPS", 1))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 900.0))

    # Model shape.
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    mask_id = vocab_size  # 1024
    total_vocab = vocab_size + 1  # 1025
    padded_vocab = int(os.environ.get("PADDED_VOCAB", 1088))  # multiple of 64 for efficiency
    num_layers = int(os.environ.get("NUM_LAYERS", 8))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    num_kv_groups = int(os.environ.get("NUM_KV_GROUPS", 4))
    mlp_mult = float(os.environ.get("MLP_MULT", 1.0))  # SwiGLU hidden = dim * mlp_mult = 512
    cond_dim = int(os.environ.get("COND_DIM", 32))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    # U-Net: num encoder layers before the bottleneck (0 = disabled)
    num_unet_layers = int(os.environ.get("NUM_UNET_LAYERS", 3))

    # Optimizer hyperparameters.
    lr = float(os.environ.get("LR", 1.1e-3))
    weight_decay = float(os.environ.get("WEIGHT_DECAY", 0.1))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 1.0))

    # LR schedule: min LR = min_lr_ratio * lr (more aggressive warmdown → better final BPB)
    min_lr_ratio = float(os.environ.get("MIN_LR_RATIO", 0.05))

    # Muon optimizer hyperparameters.
    muon_lr = float(os.environ.get("MUON_LR", 0.04))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_ns_steps = int(os.environ.get("MUON_NS_STEPS", 5))

    # Diffusion hyperparameters.
    noise_eps = float(os.environ.get("NOISE_EPS", 0.1))

    # Eval hyperparameters.
    elbo_eval_steps = int(os.environ.get("ELBO_EVAL_STEPS", 8))
    max_eval_seqs = int(os.environ.get("MAX_EVAL_SEQS", 256))
    final_eval_seqs = int(os.environ.get("FINAL_EVAL_SEQS", 4096))


# -----------------------------
# LOG-LINEAR NOISE SCHEDULE
# -----------------------------

def log_linear_noise(t: Tensor, eps: float = 1e-3) -> tuple[Tensor, Tensor]:
    """sigma(t) = -log(alpha(t)), alpha(t) = 1 - (1-eps)*t"""
    alpha = 1 - (1 - eps) * t
    sigma = -torch.log(alpha.clamp(min=1e-8))
    return sigma, alpha


# -----------------------------
# TOKENIZER-AGNOSTIC EVALUATION SETUP
# -----------------------------

def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = (tokens.numel() // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[:usable]


def count_bytes_for_tokens(
    token_ids: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> float:
    """Count total UTF-8 bytes for a token sequence (for BPB calculation)."""
    # For each token at position i, bytes = base_bytes[token_i]
    # If token_i has leading space and prev token is not boundary, add 1 byte
    flat = token_ids.reshape(-1)
    byte_counts = base_bytes_lut[flat].to(torch.float64)
    # For positions > 0, check leading space condition
    if flat.numel() > 1:
        prev = flat[:-1]
        curr = flat[1:]
        extra = (has_leading_space_lut[curr] & ~is_boundary_token_lut[prev]).to(torch.float64)
        byte_counts[1:] += extra
    return byte_counts.sum().item()


# -----------------------------
# DISCRETE ELBO EVALUATION
# -----------------------------

def eval_elbo_bpb(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    """8-point trapezoidal ELBO evaluation (discrete mask-fraction parameterization).
    NELBO per token = integral_0^1 E[CE per masked token | mask_fraction=t] dt
    No terminal KL needed: prior p(x_T) = all-MASK, which matches q(x_T|x_0) exactly.
    Returns (val_loss_nats_per_token, val_bpb).
    """
    # 8-point trapezoidal quadrature: mask fractions and weights
    _t = [0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95]
    _w = [(_t[1]-_t[0])/2,           # 0.025
          (_t[2]-_t[0])/2,           # 0.075
          (_t[3]-_t[1])/2,           # 0.125
          (_t[4]-_t[2])/2,           # 0.15
          (_t[5]-_t[3])/2,           # 0.15
          (_t[6]-_t[4])/2,           # 0.15
          (_t[7]-_t[5])/2,           # 0.15
          (_t[7]-_t[6])/2]           # 0.075
    T_EVAL = torch.tensor(_t, device=device, dtype=torch.float32)
    W_EVAL = torch.tensor(_w, device=device, dtype=torch.float64)

    seq_len = args.train_seq_len
    total_seqs = min(val_tokens.numel() // seq_len, args.max_eval_seqs)
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size

    total_elbo_nats = torch.zeros((), device=device, dtype=torch.float64)
    total_tokens = torch.zeros((), device=device, dtype=torch.float64)
    total_bytes = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    batch_size = 4
    unwrapped = model.module if hasattr(model, 'module') else model
    with torch.inference_mode():
        for batch_start in range(seq_start, seq_end, batch_size):
            batch_end = min(batch_start + batch_size, seq_end)
            bsz = batch_end - batch_start

            x0_list = []
            for s in range(batch_start, batch_end):
                start_idx = s * seq_len
                x0_list.append(val_tokens[start_idx : start_idx + seq_len])
            x0 = torch.stack(x0_list).to(device=device, dtype=torch.int64)

            for s in range(bsz):
                total_bytes += count_bytes_for_tokens(
                    x0[s], base_bytes_lut, has_leading_space_lut, is_boundary_token_lut
                )

            # 8-point quadrature: NELBO nats for this batch = Σ_k (w_k/t_k) × sum(CE × mask_k)
            seq_elbo_nats = torch.zeros(bsz, device=device, dtype=torch.float64)
            for t_val, w_val in zip(T_EVAL, W_EVAL):
                t_f = float(t_val)
                sigma_k = torch.full((bsz,), -math.log(1.0 - t_f), device=device)
                mask_k = torch.rand(bsz, seq_len, device=device) < t_f
                xt = torch.where(mask_k, args.mask_id, x0)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    log_probs = unwrapped.subs_log_probs(xt, sigma_k)

                log_p_x0 = torch.gather(log_probs.float(), -1, x0[..., None]).squeeze(-1)
                ce_masked = (-log_p_x0) * mask_k.float()
                # Contribution: w_k/t_k × sum(CE × mask) per sequence (total nats, not per token)
                step_nats = (float(w_val) / float(t_val)) * ce_masked.sum(dim=-1)
                seq_elbo_nats += step_nats.to(torch.float64)

            total_elbo_nats += seq_elbo_nats.sum()
            total_tokens += bsz * seq_len

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(total_elbo_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)

    val_loss = total_elbo_nats / total_tokens  # nats per token
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = total_tokens.item() / total_bytes.item()
    val_bpb = bits_per_token * tokens_per_byte

    model.train()
    return float(val_loss.item()), float(val_bpb)


# -----------------------------
# POST-TRAINING QUANTIZATION
# -----------------------------

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "adaln,sigma_map",
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0

def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())

def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t

def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale

def quantize_state_dict_int8(state_dict: dict[str, Tensor]):
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"),
        0,
    )
    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)
        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"] += tensor_nbytes(t)
            continue
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue
        stats["num_float_tensors"] += 1
        q, s = quantize_float_tensor(t)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)
    obj: dict[str, object] = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": quantized, "scales": scales, "dtypes": dtypes, "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats

def dequantize_state_dict_int8(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            scale = float(s.item())
            out[name] = (q.float() * scale).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


# -----------------------------
# DATA LOADING
# -----------------------------

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedSeqLoader:
    """Load batches of full sequences for diffusion training (no x/y split needed)."""
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(self, batch_size: int, seq_len: int) -> Tensor:
        """Returns (batch_size, seq_len) tensor of token ids."""
        total_tokens = batch_size * seq_len * self.world_size
        chunk = self.stream.take(total_tokens)
        # Slice for this rank
        rank_tokens = batch_size * seq_len
        start = self.rank * rank_tokens
        local = chunk[start : start + rank_tokens].to(dtype=torch.int64)
        return local.reshape(batch_size, seq_len).to(self.device, non_blocking=True)


# -----------------------------
# TRANSFORMER MODULES (BIDIRECTIONAL + AdaLN)
# -----------------------------

def rms_norm(x: Tensor) -> Tensor:
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class TimestepEmbedder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim)
        )
        half = dim // 2
        self.register_buffer(
            "freqs",
            torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half),
        )

    def forward(self, sigma: Tensor) -> Tensor:
        emb = sigma[:, None] * self.freqs[None, :]
        return self.mlp(torch.cat([emb.sin(), emb.cos()], dim=-1))


class AdaLN(nn.Module):
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, 2 * dim, bias=True)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        s, sh = self.proj(c).unsqueeze(1).chunk(2, dim=-1)
        return rms_norm(x) * (1 + s) + sh


class BidirectionalAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_groups: int = 0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_kv_heads = num_kv_groups if num_kv_groups > 0 else num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = nn.Linear(dim, dim, bias=False)
        self.c_k = nn.Linear(dim, self.kv_dim, bias=False)
        self.c_v = nn.Linear(dim, self.kv_dim, bias=False)
        self.c_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        B, T, _ = x.shape
        q = self.c_q(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = rms_norm(q)
        k = rms_norm(k)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        # Expand KV for GQA
        if self.num_kv_heads < self.num_heads:
            rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        return self.c_proj(y.transpose(1, 2).contiguous().reshape(B, T, -1))


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_mult: float, cond_dim: int, num_kv_groups: int = 0):
        super().__init__()
        self.attn = BidirectionalAttention(dim, num_heads, num_kv_groups)
        self.adaln_attn = AdaLN(dim, cond_dim)
        self.adaln_mlp = AdaLN(dim, cond_dim)
        # Fused SwiGLU: single Linear(dim, 2*hidden) for gate+up (one matmul vs two → faster)
        # hidden = dim * mlp_mult (mlp_mult=0.875 → hidden=448, saves ~745KB vs 768-hidden MLPs)
        hidden = int(dim * mlp_mult)
        self.mlp_fused = nn.Linear(dim, 2 * hidden, bias=False)
        self.mlp_proj = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, c: Tensor) -> Tensor:
        x = x + self.attn(self.adaln_attn(x, c), cos, sin)
        x_mlp = self.adaln_mlp(x, c)
        fused = self.mlp_fused(x_mlp)
        hidden = fused.shape[-1] // 2
        h = F.silu(fused[..., :hidden]) * fused[..., hidden:]
        x = x + self.mlp_proj(h)
        return x


class DiffusionLM(nn.Module):
    def __init__(self, args: Hyperparameters):
        super().__init__()
        self.args = args
        dim = args.model_dim
        self.wte = nn.Embedding(args.padded_vocab, dim)
        self.sigma_map = TimestepEmbedder(args.cond_dim)
        self.blocks = nn.ModuleList([
            Block(dim, args.num_heads, args.mlp_mult, args.cond_dim, args.num_kv_groups)
            for _ in range(args.num_layers)
        ])
        # Tied embeddings: head uses wte.weight (saves ~557K params, improves compressibility)

        # U-Net skip connections: learnable per-channel scale for each encoder→decoder pair
        self.num_unet = min(args.num_unet_layers, args.num_layers // 2)
        if self.num_unet > 0:
            self.skip_weights = nn.Parameter(torch.ones(self.num_unet, dim))

        # Precompute RoPE
        hd = dim // args.num_heads
        inv_freq = 1.0 / (args.rope_base ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))
        freqs = torch.outer(torch.arange(args.train_seq_len * 2, dtype=torch.float32), inv_freq)
        self.register_buffer("rope_cos", freqs.cos()[None, None, :, :])
        self.register_buffer("rope_sin", freqs.sin()[None, None, :, :])

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward_logits(self, xt: Tensor, sigma: Tensor) -> Tensor:
        """Raw logits for masked input xt at noise level sigma."""
        B, T = xt.shape
        x = self.wte(xt)
        c = F.silu(self.sigma_map(sigma)).to(dtype=x.dtype)
        cos = self.rope_cos[:, :, :T].to(dtype=x.dtype)
        sin = self.rope_sin[:, :, :T].to(dtype=x.dtype)

        if self.num_unet > 0:
            # Encoder pass: first num_unet layers, collect outputs for skip connections
            skips = []
            for i in range(self.num_unet):
                x = self.blocks[i](x, cos, sin, c)
                skips.append(x)
            # Middle layers (bottleneck)
            for i in range(self.num_unet, len(self.blocks) - self.num_unet):
                x = self.blocks[i](x, cos, sin, c)
            # Decoder pass: last num_unet layers, add skip-weighted encoder outputs
            for i in range(self.num_unet):
                x = x + self.skip_weights[i].to(dtype=x.dtype) * skips[self.num_unet - 1 - i]
                x = self.blocks[len(self.blocks) - self.num_unet + i](x, cos, sin, c)
        else:
            for block in self.blocks:
                x = block(x, cos, sin, c)

        logits = F.linear(rms_norm(x), self.wte.weight)[..., :self.args.total_vocab].float()
        # Logit softcap: tanh(x/cap)*cap stabilizes training, prevents runaway logits
        logits = 30.0 * torch.tanh(logits / 30.0)
        return logits

    def subs_log_probs(self, xt: Tensor, sigma: Tensor) -> Tensor:
        """MDLM substitution log probs with frozen visible tokens."""
        logits = self.forward_logits(xt, sigma)
        # Can't predict MASK token
        logits[:, :, self.args.mask_id] = -1e6
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        # Visible (unmasked) tokens: frozen to identity
        frozen = torch.full_like(logits, -1e6)
        frozen.scatter_(-1, xt[..., None], 0.0)
        visible = (xt != self.args.mask_id)[..., None]
        return torch.where(visible, frozen, logits)

    def forward(self, xt: Tensor, sigma: Tensor) -> Tensor:
        """Forward pass (calls subs_log_probs). Use this for DDP compatibility."""
        return self.subs_log_probs(xt, sigma)


# -----------------------------
# MDLM TRAINING LOSS
# -----------------------------

def mdlm_loss(model: nn.Module, x0: Tensor, args: Hyperparameters) -> Tensor:
    """Discrete mask-fraction NELBO loss for MDLM.
    Sample mask fraction t ~ Uniform(noise_eps, 1) with antithetic pairing.
    NELBO per token = E_t[CE per masked token] = integral_0^1 E[CE/t * mask] dt.
    Loss = mean_over_batch(CE * mask / t) which is an unbiased estimate of NELBO per token.
    """
    B, L = x0.shape
    eps = args.noise_eps  # 0.1 — minimum mask fraction

    # Antithetic sampling: t and (1+eps-t) are both in [eps, 1]
    t_half = torch.rand(B // 2 + 1, device=x0.device) * (1 - eps) + eps
    t = torch.cat([t_half, (1 + eps) - t_half])[:B]  # (B,)

    # Mask tokens with probability t (the mask fraction)
    mask = torch.rand(B, L, device=x0.device) < t[:, None]
    xt = torch.where(mask, args.mask_id, x0)

    # Sigma for timestep conditioning: sigma = -log(1 - t) for absorbing diffusion
    sigma = -torch.log((1 - t).clamp(min=1e-8))

    log_probs = model(xt, sigma)  # (B, L, V) via DDP wrapper for gradient sync
    log_p_x0 = torch.gather(log_probs, -1, x0[..., None]).squeeze(-1)  # (B, L)

    # ELBO weight: 1/t for masked positions (CE / t averaged over batch)
    weight = mask.float() / t[:, None].clamp(min=1e-6)
    return ((-log_p_x0) * weight).mean()


# -----------------------------
# MUON OPTIMIZER
# -----------------------------

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Newton-Schulz orthogonalization. G: (M, N) with M <= N."""
    a, b, c = 3.4445, -4.7750, 2.0315
    was_2d = G.ndim == 2
    if was_2d:
        G = G.unsqueeze(0)
    X = G.bfloat16()
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        B_mat = b * A + c * (A @ A)
        X = a * X + B_mat @ X
    if transposed:
        X = X.mT
    if was_2d:
        X = X.squeeze(0)
    return X


class Muon(torch.optim.Optimizer):
    """Muon optimizer: Nesterov momentum + Newton-Schulz orthogonalization.
    Apply to 2D weight matrices (attn/MLP). Use AdamW for embeddings/biases/norms.
    Reference: Kosson et al., "Muon: An optimizer for hidden layers in neural networks."
    """
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.bfloat16()
                state = self.state[p]
                if "buf" not in state:
                    state["buf"] = torch.zeros_like(g)
                buf = state["buf"]
                buf.mul_(momentum).add_(g)
                update = g.add(buf, alpha=momentum) if nesterov else buf.clone()
                update = zeropower_via_newtonschulz5(update, steps=ns_steps)
                scale = max(1, p.shape[-2] / p.shape[-1]) ** 0.5
                p.add_(update.to(dtype=p.dtype), alpha=-lr * scale)
        return loss


# -----------------------------
# TRAINING
# -----------------------------

def main() -> None:
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()

    # Distributed + CUDA setup
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)

    # Tokenizer + validation setup
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"val_tokens:{val_tokens.numel()}")

    # Model setup
    base_model = DiffusionLM(args).to(device).bfloat16()
    # Keep small params in fp32
    with torch.no_grad():
        for name, param in base_model.named_parameters():
            if param.ndim < 2 or "adaln" in name or "sigma_map" in name:
                param.data = param.data.float()

    base_model = torch.compile(base_model)
    model = DDP(base_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else base_model

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model:DiffusionLM(MDLM) params:{n_params}")
    log0(f"arch: {args.num_layers}L {args.model_dim}d {args.num_heads}h SwiGLU-hidden={int(args.model_dim*args.mlp_mult)}")
    log0(f"training: lr={args.lr} min_lr_ratio={args.min_lr_ratio} wd={args.weight_decay} grad_clip={args.grad_clip_norm}")
    log0(f"diffusion: noise_eps={args.noise_eps} elbo_eval_steps={args.elbo_eval_steps}")
    log0(f"batch: {args.batch_size_per_gpu}x{world_size}x{args.grad_accum_steps} seq_len={args.train_seq_len}")

    # Optimizer: Muon for 2D weight matrices, AdamW for embeddings/biases/norms/cond
    muon_params, adamw_params = [], []
    for name, p in base_model.named_parameters():
        if p.ndim >= 2 and not any(k in name for k in ("wte", "sigma_map", "adaln", "skip_weights")):
            muon_params.append(p)
        else:
            adamw_params.append(p)
    optimizer_muon = Muon(muon_params, lr=args.muon_lr, momentum=args.muon_momentum, ns_steps=args.muon_ns_steps)
    optimizer_adamw = torch.optim.AdamW(
        adamw_params,
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
        fused=True,
    )
    log0(f"optimizer: Muon({len(muon_params)} matrices lr={args.muon_lr}) + AdamW({len(adamw_params)} tensors lr={args.lr})")

    # Data loader
    train_loader = DistributedSeqLoader(args.train_files, rank, world_size, device)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def get_lr(step: int, elapsed_ms: float) -> float:
        # Warmup
        if step < args.warmup_steps:
            return args.lr * (step + 1) / args.warmup_steps
        # Warmdown based on wallclock: cosine from lr to min_lr_ratio*lr
        if max_wallclock_ms is not None:
            step_ms = elapsed_ms / max(step, 1)
            warmdown_ms = args.warmdown_iters * step_ms
            remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
            if remaining_ms <= warmdown_ms:
                progress = remaining_ms / max(warmdown_ms, 1e-9)
                min_lr = args.min_lr_ratio
                return args.lr * (min_lr + (1 - min_lr) * (0.5 * (1 + math.cos(math.pi * (1 - progress)))))
        return args.lr

    # Training loop
    training_time_ms = 0.0
    stop_after_step: int | None = None
    ema_loss = 0.0
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        # Validation
        should_validate = last_step or (args.val_loss_every > 0 and step > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_elbo_bpb(
                args, model, rank, world_size, device,
                val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms step:{step}/{args.iterations}")
            break

        # LR schedule
        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        lr = get_lr(step, elapsed_ms)
        lr_scale = lr / args.lr
        for g in optimizer_muon.param_groups:
            g["lr"] = args.muon_lr * lr_scale
        for g in optimizer_adamw.param_groups:
            g["lr"] = lr

        # Training step
        optimizer_adamw.zero_grad(set_to_none=True)
        optimizer_muon.zero_grad(set_to_none=True)
        train_loss = torch.zeros((), device=device)
        for micro_step in range(args.grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == args.grad_accum_steps - 1
            x0 = train_loader.next_batch(args.batch_size_per_gpu, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = mdlm_loss(model, x0, args) / args.grad_accum_steps
            train_loss += loss.detach() * args.grad_accum_steps
            loss.backward()
        train_loss /= args.grad_accum_steps

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        optimizer_adamw.step()
        optimizer_muon.step()

        step += 1
        tl = train_loss.item()
        ema_loss = tl if step == 1 else 0.95 * ema_loss + 0.05 * tl
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log = args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0)
        if should_log:
            log0(
                f"step:{step}/{args.iterations} train_loss:{tl:.4f} ema_loss:{ema_loss:.4f} lr:{lr:.1e} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
            )

        # Wallclock cap
        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )

    # Serialization + roundtrip validation
    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")

    quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict())
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = zlib.compress(quant_raw, level=9)
    quant_raw_bytes = len(quant_raw)
    if master_process:
        with open("final_model.int8.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize("final_model.int8.ptz")
        code_bytes = len(code.encode("utf-8"))
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
        log0(
            f"Serialized model int8+zlib: {quant_file_bytes} bytes "
            f"(payload:{quant_stats['int8_payload_bytes']} raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x)"
        )
        log0(f"Total submission size int8+zlib: {quant_file_bytes + code_bytes} bytes")

    if distributed:
        dist.barrier()
    with open("final_model.int8.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu")
    base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)
    # Final eval uses final_eval_seqs (larger than training eval) for robust estimate
    saved_max_eval = args.max_eval_seqs
    args.max_eval_seqs = args.final_eval_seqs
    log0(f"final_eval: {args.max_eval_seqs} sequences (training used {saved_max_eval})")
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_elbo_bpb(
        args, base_model, rank, world_size, device,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    log0(
        f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")
    args.max_eval_seqs = saved_max_eval

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
