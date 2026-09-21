"""Retrieval quality of the 3Di stage-2 scoring, judged by superfamily membership.

``superfamily_auroc.py`` answers this question for the sequence autoencoder, but it is
tied to that architecture: it loads the checkpoint and scores pairs by embedding cosine.
This applies the same protocol to an *alignment* score, so a scoring system can be judged
on the question a search actually asks - does a query rank its own superfamily above
everything else - rather than on how well it reproduces a reference alignment.

Every query is aligned against the whole pool, so the metrics are retrieval metrics:

* **AUROC / AUPRC** over all query-candidate pairs, with same superfamily as the label.
* **top-1**: how often the best-scoring candidate is from the query's own superfamily.
* **MAP**: mean average precision over each query's ranking, which rewards putting the
  whole superfamily near the top, not just one member.

``--negative-level topology`` restricts the negatives to candidates from the same fold or
topology but a different superfamily - the confusable case, and much harder than random
negatives. ``--randomize`` shuffles the labels as a sanity floor.

    python three_di_superfamily_auroc.py --classes cath_domain_ids.tsv --levels 4 \\
        --structures /data/cath_23M/pdb --query-profile test/profiles.npz \\
        --domains test_ids.txt --mode logodds --settings 2.1,10,1 2.1,30,3
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict

import numpy as np

from structure_alignment import SCHEMES, UnreadableStructure, align, load_chain
from structure_alignment_benchmark import read_profiles, read_target_codes


def load_classes(path: str, levels: int) -> dict[str, str]:
    out = {}
    with open(path) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) >= 2:
                out[fields[0]] = ".".join(fields[1].split(".")[:levels])
    return out


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores, kind="mergesort")
    hits = labels[order]
    if hits.sum() == 0:
        return float("nan")
    precision = np.cumsum(hits) / (np.arange(len(hits)) + 1)
    return float((precision * hits).sum() / hits.sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--classes", required=True, help="TSV of domain id -> classification")
    parser.add_argument("--levels", type=int, default=4, help="levels that define a superfamily [4]")
    parser.add_argument("--domains", required=True, help="file of held-out domain ids")
    parser.add_argument("--structures", required=True)
    parser.add_argument("--suffix", default=".pdb")
    parser.add_argument("--query-profile", required=True)
    parser.add_argument("--target-profile", default=None)
    parser.add_argument("--target-kernel", default=None)
    parser.add_argument("--profile-temperature", type=float, default=1.0)
    parser.add_argument("--profile-scale", type=int, default=100)
    parser.add_argument("--mode", default="logodds", choices=["argmax", "expected", "logodds"])
    parser.add_argument("--settings", nargs="+", required=True, metavar="W3DI,OPEN,EXTEND")
    parser.add_argument("--weight-amino", type=float, default=1.4)
    parser.add_argument("--per-family", type=int, default=4, help="pool members per superfamily [4]")
    parser.add_argument("--families", type=int, default=60, help="superfamilies in the pool [60]")
    parser.add_argument("--queries", type=int, default=120)
    parser.add_argument("--negative-level", default="any", choices=["any", "topology"],
                        help="any: all other superfamilies; topology: same parent level only")
    parser.add_argument("--min-length", type=int, default=60)
    parser.add_argument("--max-length", type=int, default=600)
    parser.add_argument("--randomize", action="store_true", help="shuffle labels, as a floor")
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    classes = load_classes(args.classes, args.levels)
    profiles = read_profiles(args.query_profile, args.profile_temperature)
    targets = (read_target_codes(args.target_profile, args.target_kernel, args.profile_temperature)
               if args.target_profile else None)
    kernel = np.load(args.target_kernel)["kernel"] if args.target_kernel else None

    # Pool: superfamilies with at least two held-out members, so a query always has a hit.
    members = defaultdict(list)
    for identifier in sorted(l.strip() for l in open(args.domains)):
        if identifier not in classes or identifier not in profiles:
            continue
        if targets is not None and identifier not in targets:
            continue
        if not os.path.exists(os.path.join(args.structures, identifier + args.suffix)):
            continue
        members[classes[identifier]].append(identifier)
    families = sorted(f for f, m in members.items() if len(m) >= 2)
    rng.shuffle(families)
    families = families[:args.families]

    chains, pool = {}, []
    for family in families:
        picked = members[family][:]
        rng.shuffle(picked)
        for identifier in picked[:args.per_family]:
            path = os.path.join(args.structures, identifier + args.suffix)
            try:
                chain = load_chain(path, name=identifier)
            except (UnreadableStructure, ValueError, KeyError):
                continue
            if not (args.min_length <= len(chain) <= args.max_length):
                continue
            if len(profiles[identifier]) != len(chain):
                continue
            if targets is not None:
                string, codes = targets[identifier]
                if len(string) != len(chain):
                    continue
                chain = type(chain)(chain.name, chain.sequence, string, chain.ca_coord, None, codes)
            chains[identifier] = chain
            pool.append(identifier)

    label = {identifier: classes[identifier] for identifier in pool}
    if args.randomize:
        shuffled = list(label.values())
        rng.shuffle(shuffled)
        label = dict(zip(label, shuffled))
    queries = [d for d in pool if sum(label[o] == label[d] for o in pool) >= 2]
    rng.shuffle(queries)
    queries = queries[:args.queries]
    print(f"pool {len(pool)} domains from {len(families)} superfamilies; {len(queries)} queries; "
          f"negatives: {args.negative_level}{'; LABELS SHUFFLED' if args.randomize else ''}")

    results = []
    for setting in args.settings:
        weight, gap_open, gap_extend = (float(x) for x in setting.split(","))
        scheme = dict(SCHEMES["3di_aa_sw"], weights=(args.weight_amino, weight),
                      gaps=(-gap_open, -gap_extend), profile=args.mode,
                      scale=args.profile_scale, target_kernel=kernel)
        all_labels, all_scores, top1, maps = [], [], [], []
        for query in queries:
            candidates = [c for c in pool if c != query]
            if args.negative_level == "topology":
                parent = ".".join(label[query].split(".")[:-1])
                candidates = [c for c in candidates
                              if label[c] == label[query] or ".".join(label[c].split(".")[:-1]) == parent]
            if sum(label[c] == label[query] for c in candidates) == 0:
                continue
            # The query carries the profile; candidates are scored as database entries.
            q = chains[query]
            q = type(q)(q.name, q.sequence, q.three_di, q.ca_coord, profiles[query], q.target_codes)
            scores = np.array([align(q, chains[c], scheme)[1] for c in candidates])
            hits = np.array([label[c] == label[query] for c in candidates], dtype=float)
            all_labels.append(hits)
            all_scores.append(scores)
            top1.append(float(hits[np.argmax(scores)]))
            maps.append(average_precision(hits, scores))
        labels = np.concatenate(all_labels)
        scores = np.concatenate(all_scores)
        order = np.argsort(scores)
        ranks = np.empty(len(order), dtype=float)
        ranks[order] = np.arange(1, len(order) + 1)
        positives, negatives = labels.sum(), len(labels) - labels.sum()
        auroc = float((ranks[labels == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))
        auprc = float(np.mean([average_precision(l, s) for l, s in zip(all_labels, all_scores)]))
        row = dict(setting=setting, auroc=auroc, map=float(np.nanmean(maps)),
                   top1=float(np.mean(top1)), pairs=int(len(labels)),
                   positive_rate=float(labels.mean()))
        results.append(row)
        print(f"   {setting:<12} AUROC {auroc:.4f}   MAP {row['map']:.4f}   "
              f"top-1 {row['top1']:.3f}   ({row['pairs']:,} pairs, {row['positive_rate']:.1%} positive)")

    if args.output:
        with open(args.output, "w") as handle:
            json.dump(dict(mode=args.mode, negative_level=args.negative_level,
                           randomized=args.randomize, pool=len(pool), queries=len(queries),
                           targets="predicted" if targets is not None else "exact",
                           results=results), handle, indent=2)


if __name__ == "__main__":
    main()
