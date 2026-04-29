"""Filter library for parameter-golf data filtering experiments.

Operates on tokenized FineWeb sp1024 shards. Doc boundaries = token id 1 (BOS).
"""
from __future__ import annotations
import argparse, glob, json, os, struct, sys, time, zlib
from collections import Counter
from pathlib import Path

import numpy as np

VOCAB = 1024
BOS_ID = 1
PAD_ID = 0
EOS_ID = 2
UNK_ID = 3
HEADER_INTS = 256
HEADER_BYTES = HEADER_INTS * 4  # 1024
SHARD_MAGIC = 20240520
SHARD_VERSION = 1


# --- I/O helpers ---

def read_shard(path: str) -> np.ndarray:
    with open(path, 'rb') as f:
        header = np.frombuffer(f.read(HEADER_BYTES), dtype='<i4').copy()
    assert int(header[0]) == SHARD_MAGIC, f'bad magic in {path}'
    assert int(header[1]) == SHARD_VERSION, f'bad version in {path}'
    n = int(header[2])
    arr = np.fromfile(path, dtype='<u2', offset=HEADER_BYTES, count=n)
    assert arr.size == n, f'short read {path}'
    return arr


def write_shard(path: str, tokens: np.ndarray) -> None:
    header = np.zeros(HEADER_INTS, dtype='<i4')
    header[0] = SHARD_MAGIC
    header[1] = SHARD_VERSION
    header[2] = len(tokens)
    with open(path, 'wb') as f:
        f.write(header.tobytes())
        f.write(tokens.astype('<u2', copy=False).tobytes())


def docs_from_shard(tokens: np.ndarray) -> list[np.ndarray]:
    """Split a flat token stream into per-doc arrays. Each doc starts with BOS=1."""
    bos_pos = np.flatnonzero(tokens == BOS_ID)
    if bos_pos.size == 0 or bos_pos[0] != 0:
        bos_pos = np.concatenate([[0], bos_pos])
    bounds = np.concatenate([bos_pos, [len(tokens)]])
    return [tokens[bounds[i]:bounds[i + 1]] for i in range(bos_pos.size)]


def concat_docs(docs: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(docs) if docs else np.zeros(0, dtype='<u2')


# --- Filters return a boolean keep-mask of length len(docs) ---

def f_length(docs, min_tokens=64, max_tokens=10_000):
    return np.array([min_tokens <= len(d) <= max_tokens for d in docs], dtype=bool)


def f_ngram_repeat(docs, n=4, frac_thr=0.20, min_doc_len=64):
    """Drop docs where the most common n-gram covers > frac_thr of all n-grams.

    Approximation: tuples of n consecutive tokens.
    """
    keep = np.ones(len(docs), dtype=bool)
    for i, d in enumerate(docs):
        if len(d) < max(min_doc_len, n + 8):
            continue  # too short to assess; let length filter handle
        # Hash n-grams to ints to count fast.
        # token id < 1024 -> fits 10 bits each; 4-gram fits in 40 bits.
        if n <= 4:
            base = np.uint64(1024)
            h = np.zeros(len(d) - n + 1, dtype=np.uint64)
            for k in range(n):
                h = h * base + d[k:k + len(h)].astype(np.uint64)
            uniq, cnt = np.unique(h, return_counts=True)
            top = int(cnt.max())
        else:
            grams = [tuple(d[j:j + n].tolist()) for j in range(len(d) - n + 1)]
            top = Counter(grams).most_common(1)[0][1]
        if top / max(1, len(d) - n + 1) > frac_thr:
            keep[i] = False
    return keep


def f_token_entropy(docs, min_ttr=0.10, min_doc_len=64):
    """Type-token ratio: |unique tokens| / |tokens|."""
    keep = np.ones(len(docs), dtype=bool)
    for i, d in enumerate(docs):
        if len(d) < min_doc_len:
            continue
        ttr = np.unique(d).size / len(d)
        if ttr < min_ttr:
            keep[i] = False
    return keep


def f_zlib_band(docs, lo=0.20, hi=0.85, min_doc_len=64):
    """Zlib compression ratio (compressed_bytes / raw_bytes). Outside band → drop."""
    keep = np.ones(len(docs), dtype=bool)
    for i, d in enumerate(docs):
        if len(d) < min_doc_len:
            continue
        raw = d.astype('<u2', copy=False).tobytes()
        comp = zlib.compress(raw, 6)
        ratio = len(comp) / max(1, len(raw))
        if not (lo <= ratio <= hi):
            keep[i] = False
    return keep


# --- Filters needing detokenization ---

def f_alpha_ratio(docs, sp, min_alpha=0.70, min_doc_len=64):
    keep = np.ones(len(docs), dtype=bool)
    for i, d in enumerate(docs):
        if len(d) < min_doc_len:
            continue
        text = sp.decode([int(t) for t in d if t > 3])
        if not text:
            keep[i] = False
            continue
        alpha = sum(1 for c in text if c.isalpha())
        if alpha / len(text) < min_alpha:
            keep[i] = False
    return keep


_STOPWORDS = set("the a an and or but if of for to in on at by with as is are was were be been being have has had do does did this that these those i you he she it we they not from".split())


def f_stopword_ratio(docs, sp, min_stop=0.05, min_doc_len=64):
    keep = np.ones(len(docs), dtype=bool)
    for i, d in enumerate(docs):
        if len(d) < min_doc_len:
            continue
        text = sp.decode([int(t) for t in d if t > 3]).lower()
        words = text.split()
        if not words:
            keep[i] = False
            continue
        stop = sum(1 for w in words if w in _STOPWORDS)
        if stop / len(words) < min_stop:
            keep[i] = False
    return keep


# --- Self-perplexity scoring uses the baseline checkpoint ---
# Implemented in score_self_ppl.py (separate, GPU-bound).


# --- Driver: apply a named filter to all train shards ---

NAMED_FILTERS = {
    'length':        lambda docs, sp: f_length(docs, min_tokens=128),
    'ngram_rep':     lambda docs, sp: f_ngram_repeat(docs, n=4, frac_thr=0.15),
    'entropy':       lambda docs, sp: f_token_entropy(docs, min_ttr=0.12),
    'zlib_band':     lambda docs, sp: f_zlib_band(docs, lo=0.25, hi=0.80),
    'alpha':         lambda docs, sp: f_alpha_ratio(docs, sp, min_alpha=0.70),
    'stopword':      lambda docs, sp: f_stopword_ratio(docs, sp, min_stop=0.05),
    'stack_cheap':   lambda docs, sp: (
        f_length(docs, 128) & f_ngram_repeat(docs, n=4, frac_thr=0.15)
        & f_token_entropy(docs, 0.12) & f_zlib_band(docs, lo=0.25, hi=0.80)
    ),
    'stack_full':    lambda docs, sp: (
        f_length(docs, 128) & f_ngram_repeat(docs, n=4, frac_thr=0.15)
        & f_token_entropy(docs, 0.12) & f_zlib_band(docs, lo=0.25, hi=0.80)
        & f_alpha_ratio(docs, sp, 0.70) & f_stopword_ratio(docs, sp, 0.05)
    ),
}


def apply_filter(name: str, src_dir: str, dst_dir: str,
                 sp_path: str | None = None) -> dict:
    if name not in NAMED_FILTERS:
        raise SystemExit(f'unknown filter: {name}; have {list(NAMED_FILTERS)}')
    fn = NAMED_FILTERS[name]
    sp = None
    if sp_path is not None:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.load(sp_path)

    os.makedirs(dst_dir, exist_ok=True)
    # symlink val unchanged
    for vp in sorted(glob.glob(os.path.join(src_dir, 'fineweb_val_*.bin'))):
        dst_v = os.path.join(dst_dir, os.path.basename(vp))
        if os.path.exists(dst_v) or os.path.islink(dst_v):
            os.remove(dst_v)
        os.symlink(vp, dst_v)

    train_files = sorted(glob.glob(os.path.join(src_dir, 'fineweb_train_*.bin')))
    stats = {'name': name, 'shards': [], 'total_in': 0, 'total_out': 0,
             'docs_in': 0, 'docs_out': 0, 't_start': time.time()}

    for tp in train_files:
        t0 = time.time()
        toks = read_shard(tp)
        docs = docs_from_shard(toks)
        keep = fn(docs, sp)
        kept = [d for d, k in zip(docs, keep) if k]
        out_toks = concat_docs(kept)
        out_path = os.path.join(dst_dir, os.path.basename(tp))
        write_shard(out_path, out_toks)
        s = {'shard': os.path.basename(tp), 'in_tok': int(toks.size),
             'out_tok': int(out_toks.size), 'in_doc': len(docs),
             'out_doc': int(keep.sum()), 'sec': round(time.time() - t0, 2)}
        stats['shards'].append(s)
        stats['total_in'] += s['in_tok']
        stats['total_out'] += s['out_tok']
        stats['docs_in'] += s['in_doc']
        stats['docs_out'] += s['out_doc']
        print(f'[{name}] {s["shard"]}: docs {s["in_doc"]} -> {s["out_doc"]} '
              f'({s["out_doc"]/max(1,s["in_doc"]):.2%}); '
              f'tokens {s["in_tok"]:,} -> {s["out_tok"]:,} '
              f'({s["out_tok"]/max(1,s["in_tok"]):.2%}); {s["sec"]}s', flush=True)
    stats['elapsed'] = round(time.time() - stats['t_start'], 2)
    stats['keep_doc_pct'] = stats['docs_out'] / max(1, stats['docs_in'])
    stats['keep_tok_pct'] = stats['total_out'] / max(1, stats['total_in'])

    with open(os.path.join(dst_dir, 'filter_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True, choices=list(NAMED_FILTERS))
    ap.add_argument('--src', default='/workspace/parameter-golf/data/datasets/fineweb10B_sp1024')
    ap.add_argument('--dst', default=None)
    ap.add_argument('--sp', default='/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model')
    args = ap.parse_args()
    dst = args.dst or f'{args.src}_{args.name}'
    s = apply_filter(args.name, args.src, dst, args.sp)
    print('SUMMARY:', json.dumps({k: v for k, v in s.items() if k != 'shards'}, indent=2))


if __name__ == '__main__':
    main()
