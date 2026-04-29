"""Build a filtered dataset by per-doc score thresholds.

Reads /workspace/scores/<shard>.npy and the original shards; emits a new dir.
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time
import numpy as np

sys.path.insert(0, '/workspace/parameter-golf')
import filter_lib as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='/workspace/parameter-golf/data/datasets/fineweb10B_sp1024')
    ap.add_argument('--scores', default='/workspace/scores')
    ap.add_argument('--dst', required=True)
    ap.add_argument('--mode', required=True,
                    choices=['drop_high', 'drop_low', 'band', 'keep_high', 'keep_low'])
    ap.add_argument('--q', type=float, default=0.10,
                    help='quantile fraction to drop (or band half-width)')
    ap.add_argument('--lo', type=float, default=0.20)
    ap.add_argument('--hi', type=float, default=0.80)
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    # symlink val
    for vp in sorted(glob.glob(os.path.join(args.src, 'fineweb_val_*.bin'))):
        dv = os.path.join(args.dst, os.path.basename(vp))
        if os.path.lexists(dv):
            os.remove(dv)
        os.symlink(vp, dv)

    # Pool ALL scores to compute global thresholds (consistent across shards).
    train_files = sorted(glob.glob(os.path.join(args.src, 'fineweb_train_*.bin')))
    all_scores = []
    per_shard_docs = []
    for tp in train_files:
        toks = F.read_shard(tp)
        docs = F.docs_from_shard(toks)
        per_shard_docs.append(docs)
        sp = os.path.join(args.scores, os.path.basename(tp).replace('.bin', '.npy'))
        scores = np.load(sp)
        # NaN docs are too short — give them a sentinel that excludes them from quantile
        all_scores.append(scores)
    flat = np.concatenate(all_scores)
    valid = ~np.isnan(flat)
    vals = flat[valid]

    if args.mode in ('drop_high', 'keep_low'):
        cut = np.quantile(vals, 1 - args.q)
        decide = lambda s: ~np.isnan(s) & (s <= cut)
        thresh_desc = f'cut_high={cut:.4f}'
    elif args.mode in ('drop_low', 'keep_high'):
        cut = np.quantile(vals, args.q)
        decide = lambda s: ~np.isnan(s) & (s >= cut)
        thresh_desc = f'cut_low={cut:.4f}'
    elif args.mode == 'band':
        cut_lo = np.quantile(vals, args.lo)
        cut_hi = np.quantile(vals, args.hi)
        decide = lambda s: ~np.isnan(s) & (s >= cut_lo) & (s <= cut_hi)
        thresh_desc = f'band={cut_lo:.4f}..{cut_hi:.4f}'
    else:
        raise ValueError(args.mode)

    stats = {'mode': args.mode, 'q': args.q, 'lo': args.lo, 'hi': args.hi,
             'thresh_desc': thresh_desc,
             'shards': [], 'docs_in': 0, 'docs_out': 0,
             'tok_in': 0, 'tok_out': 0, 't_start': time.time()}

    for tp, docs, scores in zip(train_files, per_shard_docs, all_scores):
        keep = decide(scores)
        kept = [d for d, k in zip(docs, keep) if k]
        out_toks = F.concat_docs(kept)
        out_path = os.path.join(args.dst, os.path.basename(tp))
        F.write_shard(out_path, out_toks)
        in_tok = sum(len(d) for d in docs)
        out_tok = int(out_toks.size)
        s = {'shard': os.path.basename(tp), 'in_doc': len(docs),
             'out_doc': int(keep.sum()), 'in_tok': in_tok, 'out_tok': out_tok}
        stats['shards'].append(s)
        stats['docs_in'] += s['in_doc']; stats['docs_out'] += s['out_doc']
        stats['tok_in'] += s['in_tok']; stats['tok_out'] += s['out_tok']
        print(f'[{args.mode}] {s["shard"]}: docs {s["in_doc"]} -> {s["out_doc"]} '
              f'({s["out_doc"]/max(1,s["in_doc"]):.2%}); '
              f'tokens {in_tok:,} -> {out_tok:,} ({out_tok/max(1,in_tok):.2%})', flush=True)
    stats['elapsed'] = round(time.time() - stats['t_start'], 1)
    stats['keep_doc_pct'] = stats['docs_out'] / max(1, stats['docs_in'])
    stats['keep_tok_pct'] = stats['tok_out'] / max(1, stats['tok_in'])
    with open(os.path.join(args.dst, 'filter_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    print('SUMMARY:', json.dumps({k: v for k, v in stats.items() if k != 'shards'}, indent=2))


if __name__ == '__main__':
    main()
