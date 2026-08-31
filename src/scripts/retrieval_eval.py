"""Measure the model as a first-stage filter: recall against candidate-list size.

AUROC and spearman rank every pair equally, but a filter is only ever used at
one operating point — the candidate list handed to full pairwise alignment.  The
question that matters is therefore *how many candidates must stage 2 align to
catch 95% of true homologs*, and neither training metrics nor AUROC answer it.

Retrieval is segment-level, matching how such an index is actually queried: a
domain is a hit when **any** of its windows is close to any window of the query
(the `max` read-out).  Ground truth is CATH homologous-superfamily membership,
which the model never trains on.

Report includes a random-retrieval column (recall@k = k/N), because with a few
large superfamilies in the database even blind retrieval scores non-trivially.
"""
from __future__ import annotations

import argparse
import logging
import random
from collections import defaultdict

import numpy as np
import torch

from dataset.utils.cath_superfamily import load_superfamilies, parse_cath_domain_id
from scripts.superfamily_auroc import encode, load_model, read_domains, windows_of

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def domain_scores(emb: torch.Tensor, bounds: np.ndarray, chunk: int) -> np.ndarray:
    """(n_domains, n_domains) max cosine between any window pair of each domain.

    ``reduceat`` collapses each domain's contiguous block of windows in one pass,
    first over the database axis then over the query axis, which avoids
    materialising the full window-by-window matrix for the whole corpus.
    """
    e = emb.numpy().astype(np.float32)
    n = len(bounds) - 1
    out = np.empty((n, n), dtype=np.float32)
    starts = bounds[:-1]
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        block = e[bounds[lo]:bounds[hi]] @ e.T                  # (windows, all windows)
        block = np.maximum.reduceat(block, starts, axis=1)      # -> (windows, domains)
        rows = starts[lo:hi] - bounds[lo]
        out[lo:hi] = np.maximum.reduceat(block, rows, axis=0)   # -> (domains, domains)
        logger.info(f"    scored {hi:,}/{n:,} query domains")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--fasta', required=True)
    ap.add_argument('--domain-list', required=True)
    ap.add_argument('--domains', default=None, help='holdout file')
    ap.add_argument('--segment-length', type=int, default=50)
    ap.add_argument('--windows-per-domain', type=int, default=4)
    ap.add_argument('--max-domains', type=int, default=3000)
    ap.add_argument('--per-family', type=int, default=12)
    ap.add_argument('--nhead', type=int, default=12)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--chunk', type=int, default=200)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--randomize', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    device = torch.device(
        ('cuda' if torch.cuda.is_available() else
         'mps' if torch.backends.mps.is_available() else 'cpu')
        if args.device == 'auto' else args.device)
    rng = random.Random(args.seed)

    keep = None
    if args.domains:
        with open(args.domains) as h:
            keep = {line.split()[0] for line in h if line.strip()}

    superfamily = load_superfamilies(args.domain_list)
    sequences = read_domains(args.fasta, keep, args.segment_length)
    families = defaultdict(list)
    for d in sequences:
        if d in superfamily:
            families[superfamily[d]].append(d)
    families = {f: m for f, m in families.items() if len(m) >= 2}

    order = list(families)
    rng.shuffle(order)
    chosen: list[str] = []
    for f in order:
        if len(chosen) >= args.max_domains:
            break
        m = families[f]
        chosen.extend(m if len(m) <= args.per_family else rng.sample(m, args.per_family))
    label = np.array([superfamily[d] for d in chosen])
    logger.info(f"database: {len(chosen):,} domains, {len(set(label)):,} superfamilies")

    segments, bounds = [], [0]
    for d in chosen:
        segments.extend(windows_of(sequences[d], args.segment_length, args.windows_per_domain))
        bounds.append(len(segments))
    bounds = np.asarray(bounds)
    logger.info(f"encoding {len(segments):,} windows on {device} ...")
    emb = encode(load_model(args.checkpoint, args.nhead, device, args.randomize),
                 segments, device, args.batch_size)

    logger.info("scoring domain pairs ...")
    scores = domain_scores(emb, bounds, args.chunk)
    np.fill_diagonal(scores, -np.inf)                    # never retrieve the query itself

    n = len(chosen)
    same = label[:, None] == label[None, :]
    np.fill_diagonal(same, False)
    n_pos = same.sum(1)
    usable = n_pos > 0
    ranked = np.argsort(-scores, axis=1)
    hits = np.take_along_axis(same, ranked, axis=1)      # (queries, ranked db) boolean
    cum = np.cumsum(hits, axis=1)

    ks = [k for k in (1, 5, 10, 25, 50, 100, 250, 500, 1000, 2000) if k < n]
    print("\n" + "=" * 74)
    print(f"  {n:,} domains in the database; {usable.sum():,} queries have >=1 homolog "
          f"(median {int(np.median(n_pos[usable]))} homologs each)")
    print(f"\n  {'k':>6s}{'recall@k':>11s}{'random':>10s}{'% of db':>10s}")
    recall_at = {}
    for k in ks:
        r = float(np.mean(cum[usable, k - 1] / n_pos[usable]))
        recall_at[k] = r
        print(f"  {k:6d}{r:11.4f}{k / n:10.4f}{100 * k / n:9.1f}%")

    print()
    for target in (0.90, 0.95, 0.99):
        per_query = cum[usable] / n_pos[usable][:, None]
        reached = (per_query >= target).argmax(axis=1) + 1
        reached = np.where(per_query[:, -1] >= target, reached, n)
        k_med = int(np.median(reached))
        k_mean_curve = next((k for k in range(1, n) if
                             float(np.mean(cum[usable, k - 1] / n_pos[usable])) >= target), None)
        print(f"  {int(target * 100)}% recall: median query needs k={k_med:,} "
              f"({100 * k_med / n:.1f}% of db)"
              + (f" | mean-recall curve crosses at k={k_mean_curve:,} ({100 * k_mean_curve / n:.1f}%)"
                 if k_mean_curve else " | mean recall never reaches this"))
    print(f"\n  'k' is how many candidates stage 2 must align per query. With a median of "
          f"{int(np.median(n_pos[usable]))} homologs per query, per-query recall moves in steps of "
          f"1/{int(np.median(n_pos[usable]))}, so nearby targets can coincide; the mean-recall "
          f"curve is the smoother number for capacity planning.")


if __name__ == '__main__':
    main()
