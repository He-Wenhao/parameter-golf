"""Score per-doc self-perplexity using the contest baseline checkpoint.

Uses train_gpt.GPT loaded from /workspace/parameter-golf/final_model.pt (full precision).
Outputs /workspace/scores/<shard>.npy: float32 array of mean per-token CE loss per doc.
"""
from __future__ import annotations
import argparse, glob, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '/workspace/parameter-golf')
import train_gpt as TG  # provides GPT class
import filter_lib as F_LIB

DEVICE = 'cuda'
MAX_DOC_LEN = 2048      # truncate long docs to keep batches sane
BATCH_TOKENS = 8192     # ~ tokens per micro-batch (sum of doc lengths)


def build_model() -> 'TG.GPT':
    m = TG.GPT(
        vocab_size=1024, num_layers=9, model_dim=512, num_heads=8,
        num_kv_heads=4, mlp_mult=2, tie_embeddings=True,
        tied_embed_init_std=0.005, logit_softcap=30.0,
        rope_base=10000.0, qk_gain_init=1.5,
    )
    sd = torch.load('/workspace/parameter-golf/final_model.pt',
                    map_location='cpu', weights_only=True)
    m.load_state_dict(sd, strict=True)
    m.eval().to(DEVICE).to(torch.bfloat16)
    return m


@torch.no_grad()
def score_doc_batch(model, doc_tokens_list):
    """For each doc tensor (1D), return mean per-token CE loss (skipping first token).

    Pads to max length, masks out pad targets with -100.
    """
    device = DEVICE
    L = max(len(d) for d in doc_tokens_list)
    if L < 2:
        return [float('nan')] * len(doc_tokens_list)
    B = len(doc_tokens_list)
    inp = torch.zeros((B, L - 1), dtype=torch.long, device=device)
    tgt = torch.full((B, L - 1), -100, dtype=torch.long, device=device)
    real_lens = []
    for i, d in enumerate(doc_tokens_list):
        n = len(d)
        if n < 2:
            real_lens.append(0); continue
        t = torch.from_numpy(d.astype(np.int64)).to(device)
        inp[i, :n - 1] = t[:-1]
        tgt[i, :n - 1] = t[1:]
        real_lens.append(n - 1)
    # Use the model's forward by extracting logits manually:
    x = model.tok_emb(inp)
    x = F.rms_norm(x, (x.size(-1),))
    x0 = x
    skips = []
    for i in range(model.num_encoder_layers):
        x = model.blocks[i](x, x0); skips.append(x)
    for i in range(model.num_decoder_layers):
        if skips:
            x = x + model.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
        x = model.blocks[model.num_encoder_layers + i](x, x0)
    x = model.final_norm(x)
    logits = F.linear(x, model.tok_emb.weight)
    logits = (model.logit_softcap * torch.tanh(logits / model.logit_softcap)).float()
    losses = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
        reduction='none', ignore_index=-100
    ).reshape(B, -1)
    out = []
    for i, n in enumerate(real_lens):
        if n == 0:
            out.append(float('nan')); continue
        out.append(losses[i, :n].mean().item())
    return out


def score_shard(model, shard_path: str, out_path: str):
    toks = F_LIB.read_shard(shard_path)
    docs = F_LIB.docs_from_shard(toks)
    # Optionally truncate very long docs
    docs_trunc = [d[:MAX_DOC_LEN] for d in docs]
    scores = np.full(len(docs), np.nan, dtype=np.float32)

    # Build batches by greedy bin-packing on doc length.
    order = sorted(range(len(docs_trunc)), key=lambda i: len(docs_trunc[i]))
    cur, cur_idx, cur_max = [], [], 0
    batches = []
    for i in order:
        ln = len(docs_trunc[i])
        new_max = max(cur_max, ln)
        if cur and (len(cur) + 1) * new_max > BATCH_TOKENS:
            batches.append(cur_idx)
            cur, cur_idx, cur_max = [], [], 0
        cur.append(docs_trunc[i]); cur_idx.append(i); cur_max = max(cur_max, ln)
    if cur_idx:
        batches.append(cur_idx)

    t0 = time.time()
    for bi, idxs in enumerate(batches):
        bt = [docs_trunc[i] for i in idxs]
        out = score_doc_batch(model, bt)
        for j, i in enumerate(idxs):
            scores[i] = out[j]
        if (bi + 1) % 100 == 0:
            print(f'  {os.path.basename(shard_path)} batch {bi + 1}/{len(batches)} '
                  f'({time.time() - t0:.1f}s)', flush=True)
    np.save(out_path, scores)
    print(f'  saved {out_path}; mean_score={np.nanmean(scores):.4f}; t={time.time() - t0:.1f}s')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='/workspace/parameter-golf/data/datasets/fineweb10B_sp1024')
    ap.add_argument('--out', default='/workspace/scores')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    model = build_model()
    print('model loaded', flush=True)
    for sp in sorted(glob.glob(os.path.join(args.src, 'fineweb_train_*.bin'))):
        out = os.path.join(args.out, os.path.basename(sp).replace('.bin', '.npy'))
        if os.path.exists(out):
            print(f'skip {sp} (exists)'); continue
        print(f'scoring {sp}', flush=True)
        score_shard(model, sp, out)


if __name__ == '__main__':
    main()
