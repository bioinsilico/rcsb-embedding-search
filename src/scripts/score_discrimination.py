"""How well does a scoring system separate related from unrelated pairs?

The alignment benchmark only ever scores pairs that are already related (TM >= 0.7): it
asks whether the residues are aligned correctly, never whether an unrelated pair would
score highly.  A search engine needs both.  Parameters that maximize alignment quality
can make gaps so cheap, relative to the per-residue score, that unrelated pairs also
build long high-scoring alignments - which destroys E-values and the filters built on
them, even though the benchmark looks better.

This scores two populations with the same scoring system - related pairs (TM >= 0.7) and
unrelated pairs (different fold) - and reports how separable they are:

* AUC, the probability that a related pair outscores an unrelated one;
* the true-positive rate at a 1% and 0.1% false-positive rate;
* the raw score distributions, since length-normalized behaviour matters too.

    python score_discrimination.py --related cath_val_pairs.csv --unrelated cath_val_unrelated.csv \\
        --structures /data/cath_23M/pdb --query-profile val/profiles.npz \\
        --mode logodds --settings 2.1,10,1 3.0,30,3
"""
from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np

from structure_alignment import SCHEMES, UnreadableStructure, align, load_chain
from structure_alignment_benchmark import read_profiles, read_target_codes, sample_pairs


def collect(path, structures, suffix, profiles, targets, limit, min_length, max_length, seed):
    """Load pairs once; returns chains ready to align."""
    rng = random.Random(seed)
    out, cache = [], {}
    for identifier_a, identifier_b, _ in sample_pairs(path, 0.0, limit * 6, rng):
        if len(out) >= limit:
            break
        if identifier_a not in profiles or (targets is not None and identifier_b not in targets):
            continue
        paths = [os.path.join(structures, i + suffix) for i in (identifier_a, identifier_b)]
        if not all(os.path.exists(p) for p in paths):
            continue
        try:
            for identifier, p in zip((identifier_a, identifier_b), paths):
                if identifier not in cache:
                    cache[identifier] = load_chain(p, name=identifier)
        except (UnreadableStructure, ValueError, KeyError):
            continue
        query, target = cache[identifier_a], cache[identifier_b]
        if not (min_length <= len(query) <= max_length and min_length <= len(target) <= max_length):
            continue
        profile = profiles[identifier_a]
        if len(profile) != len(query):
            continue
        query = type(query)(query.name, query.sequence, query.three_di, query.ca_coord, profile, None)
        if targets is not None:
            string, codes = targets[identifier_b]
            if len(string) != len(target):
                continue
            target = type(target)(target.name, target.sequence, string, target.ca_coord, None, codes)
        out.append((query, target))
    return out


def auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """Probability a related pair outscores an unrelated one (ties count a half)."""
    order = np.argsort(np.concatenate([positive, negative]), kind="mergesort")
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    values = np.concatenate([positive, negative])
    # average ranks over ties
    for value in np.unique(values):
        tied = values == value
        if tied.sum() > 1:
            ranks[tied] = ranks[tied].mean()
    rank_sum = ranks[:len(positive)].sum()
    return float((rank_sum - len(positive) * (len(positive) + 1) / 2) / (len(positive) * len(negative)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--related", required=True)
    parser.add_argument("--unrelated", required=True)
    parser.add_argument("--structures", required=True)
    parser.add_argument("--suffix", default=".pdb")
    parser.add_argument("--query-profile", required=True)
    parser.add_argument("--target-profile", default=None)
    parser.add_argument("--target-kernel", default=None)
    parser.add_argument("--profile-temperature", type=float, default=1.0)
    parser.add_argument("--profile-scale", type=int, default=100)
    parser.add_argument("--mode", default="logodds", choices=["argmax", "expected", "logodds"])
    parser.add_argument("--settings", nargs="+", required=True, metavar="W3DI,OPEN,EXTEND",
                        help="scoring settings to compare, e.g. 2.1,10,1 3.0,30,3")
    parser.add_argument("--weight-amino", type=float, default=1.4)
    parser.add_argument("--limit", type=int, default=400, help="pairs per population")
    parser.add_argument("--min-length", type=int, default=60)
    parser.add_argument("--max-length", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    profiles = read_profiles(args.query_profile, args.profile_temperature)
    targets = (read_target_codes(args.target_profile, args.target_kernel, args.profile_temperature)
               if args.target_profile else None)
    kernel = np.load(args.target_kernel)["kernel"] if args.target_kernel else None

    populations = {
        name: collect(path, args.structures, args.suffix, profiles, targets, args.limit,
                      args.min_length, args.max_length, args.seed)
        for name, path in (("related", args.related), ("unrelated", args.unrelated))
    }
    print(f"{len(populations['related'])} related and {len(populations['unrelated'])} unrelated pairs; "
          f"mode {args.mode}, targets {'predicted' if targets is not None else 'exact'}")

    results = []
    print(f"\n{'w3di/open/extend':<18}{'AUC':>7}{'TPR@1%FPR':>11}{'TPR@0.1%':>10}   "
          f"{'related median':>15}{'unrelated median':>18}")
    for setting in args.settings:
        weight, gap_open, gap_extend = (float(x) for x in setting.split(","))
        scheme = dict(SCHEMES["3di_aa_sw"], weights=(args.weight_amino, weight),
                      gaps=(-gap_open, -gap_extend), profile=args.mode,
                      scale=args.profile_scale, target_kernel=kernel)
        scores = {name: np.array([align(q, t, scheme)[1] for q, t in pairs])
                  for name, pairs in populations.items()}
        area = auc(scores["related"], scores["unrelated"])
        cut_1 = np.quantile(scores["unrelated"], 0.99)
        cut_01 = np.quantile(scores["unrelated"], 0.999)
        row = dict(weight_3di=weight, gap_open=gap_open, gap_extend=gap_extend, auc=area,
                   tpr_at_1pct=float((scores["related"] > cut_1).mean()),
                   tpr_at_0p1pct=float((scores["related"] > cut_01).mean()),
                   related_median=float(np.median(scores["related"])),
                   unrelated_median=float(np.median(scores["unrelated"])),
                   related=[float(x) for x in scores["related"]],
                   unrelated=[float(x) for x in scores["unrelated"]])
        results.append(row)
        print(f"{setting:<18}{area:>7.4f}{row['tpr_at_1pct']:>11.3f}{row['tpr_at_0p1pct']:>10.3f}   "
              f"{row['related_median']:>15.1f}{row['unrelated_median']:>18.1f}")

    if args.output:
        with open(args.output, "w") as handle:
            json.dump(dict(mode=args.mode, targets="predicted" if targets is not None else "exact",
                           results=results), handle)


if __name__ == "__main__":
    main()
