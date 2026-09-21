"""Estimate Karlin-Altschul lambda and K for a 3Di+amino-acid scoring system.

FoldMatch's E-values come from constants calibrated for BLOSUM62 with gaps 11/1
(``search/karlin_altschul.py``).  Adding a 3Di term, and retuning the weight and gaps,
makes a different scoring system, so those constants no longer describe it: E-values
would be wrong by orders of magnitude, and every filter keyed on them with it.

The estimate here is the textbook one.  For local alignment of unrelated sequences the
optimal score follows an extreme-value distribution,

    P(S >= x) = 1 - exp(-K m n exp(-lambda x))

so scoring unrelated pairs and fitting that distribution gives lambda and K.

The null matters.  ``--null unrelated`` (the default) uses real domains from different
folds, which keeps the run structure of both alphabets - 3Di is far more skewed than
amino acids, ``v`` alone being 21% of residues, and a shuffled target loses exactly the
runs that a real one has.  ``--null shuffle`` instead permutes each target's residues,
which is cheaper but gives a heavier tail and a badly biased lambda.

Only the tail is fitted, as is standard for extreme-value fits: the bulk of the
distribution is not Gumbel, and it is the tail that E-values are read from.  Points
below ``--tail-quantile`` are treated as censored, so they contribute only the
probability of being below the threshold, not a likelihood of their own.

It also reports, for a soft (position-specific) query, how much lambda varies from query
to query: a profile is a different scoring system per query, as it is for PSI-BLAST, so
one global lambda may not be enough.

    python calibrate_evalue.py --pairs cath_val_pairs.csv --structures /data/cath_23M/pdb \\
        --query-profile val/profiles.npz --mode logodds --weight-3di 3.0 --gap-open 30 \\
        --gap-extend 3 --limit 1200
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random

import numpy as np

from structure_alignment import (
    AMINO_ACIDS, SCHEMES, THREE_DI, UnreadableStructure, align, load_chain, profile_term,
    _base_scores,
)
from structure_alignment_benchmark import read_profiles, read_target_codes, sample_pairs


def fit_lambda_k(scores: np.ndarray, area: np.ndarray, tail_quantile: float = 0.0) -> tuple[float, float]:
    """Maximum-likelihood (lambda, K) for ``P(S >= x) = 1 - exp(-K*area*exp(-lambda x))``.

    With ``tail_quantile > 0`` the fit is censored at that quantile of the scores: points
    above it contribute their density, points below only the probability of being below,
    so the bulk of the distribution cannot drag the tail parameters.
    """
    from scipy.optimize import minimize

    threshold = float(np.quantile(scores, tail_quantile)) if tail_quantile > 0 else -np.inf
    above = scores >= threshold

    def negative_log_likelihood(parameters: np.ndarray) -> float:
        lam, log_k = parameters
        if lam <= 0:
            return 1e12
        # density of the EVD at the observed scores above the threshold
        z = lam * scores[above] - np.log(area[above]) - log_k
        value = -np.sum(np.log(lam) - z - np.exp(-z))
        if np.isfinite(threshold):
            # censored points: log P(S < threshold) = -K*area*exp(-lambda*threshold)
            censored = np.exp(log_k + np.log(area[~above]) - lam * threshold)
            value += float(np.sum(censored))
        return float(value)

    best, best_value = None, np.inf
    for lam0 in (0.05, 0.1, 0.2, 0.4, 0.8):
        start = np.array([lam0, math.log(1e-3)])
        result = minimize(negative_log_likelihood, start, method="Nelder-Mead",
                          options=dict(maxiter=4000, xatol=1e-6, fatol=1e-6))
        if result.fun < best_value:
            best, best_value = result.x, result.fun
    return float(best[0]), float(math.exp(best[1]))


def analytic_lambda(rows: np.ndarray, background: np.ndarray) -> float:
    """lambda solving ``mean_i sum_c bg(c) exp(lambda * row_i[c]) = 1`` for a position-specific matrix."""
    def excess(lam: float) -> float:
        return float(np.mean(np.exp(lam * rows) @ background) - 1.0)

    low, high = 1e-4, 5.0
    if excess(high) < 0:
        return float("nan")
    for _ in range(200):
        mid = (low + high) / 2
        if excess(mid) > 0:
            high = mid
        else:
            low = mid
    return (low + high) / 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--structures", required=True)
    parser.add_argument("--suffix", default=".pdb")
    parser.add_argument("--query-profile", required=True)
    parser.add_argument("--target-profile", default=None)
    parser.add_argument("--target-kernel", default=None)
    parser.add_argument("--profile-temperature", type=float, default=1.0)
    parser.add_argument("--profile-scale", type=int, default=100)
    parser.add_argument("--mode", default="logodds", choices=["argmax", "expected", "logodds"])
    parser.add_argument("--weight-amino", type=float, default=1.4)
    parser.add_argument("--weight-3di", type=float, default=2.1)
    parser.add_argument("--gap-open", type=float, default=10)
    parser.add_argument("--gap-extend", type=float, default=1)
    parser.add_argument("--limit", type=int, default=1000, help="unrelated pairs to score")
    parser.add_argument("--null", default="unrelated", choices=["unrelated", "shuffle"],
                        help="unrelated: use the pairs as given (different folds); "
                             "shuffle: permute each target's residues [unrelated]")
    parser.add_argument("--tail-quantile", type=float, default=0.75,
                        help="fit only scores above this quantile, censoring the rest [0.75]")
    parser.add_argument("--min-length", type=int, default=60)
    parser.add_argument("--max-length", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    numpy_rng = np.random.default_rng(args.seed)
    profiles = read_profiles(args.query_profile, args.profile_temperature)
    targets = (read_target_codes(args.target_profile, args.target_kernel, args.profile_temperature)
               if args.target_profile else None)
    kernel = np.load(args.target_kernel)["kernel"] if args.target_kernel else None
    scheme = dict(SCHEMES["3di_aa_sw"], weights=(args.weight_amino, args.weight_3di),
                  gaps=(-args.gap_open, -args.gap_extend), profile=args.mode,
                  scale=args.profile_scale, target_kernel=kernel)

    candidates = sample_pairs(args.pairs, 0.0, args.limit * 6, rng)
    cache, rows, lengths, per_query = {}, [], [], {}
    symbol_counts = None
    for identifier_a, identifier_b, _ in candidates:
        if len(rows) >= args.limit:
            break
        if identifier_a not in profiles or (targets is not None and identifier_b not in targets):
            continue
        paths = [os.path.join(args.structures, i + args.suffix) for i in (identifier_a, identifier_b)]
        if not all(os.path.exists(p) for p in paths):
            continue
        try:
            for identifier, path in zip((identifier_a, identifier_b), paths):
                if identifier not in cache:
                    cache[identifier] = load_chain(path, name=identifier)
        except (UnreadableStructure, ValueError, KeyError):
            continue
        query, target = cache[identifier_a], cache[identifier_b]
        if not (args.min_length <= len(query) <= args.max_length
                and args.min_length <= len(target) <= args.max_length):
            continue
        profile = profiles[identifier_a]
        if len(profile) != len(query):
            continue
        query = type(query)(query.name, query.sequence, query.three_di, query.ca_coord, profile, None)
        codes = None
        if targets is not None:
            string, codes = targets[identifier_b]
            if len(string) != len(target):
                continue
            target = type(target)(target.name, target.sequence, string, target.ca_coord, None, codes)

        if args.null == "shuffle":
            order = numpy_rng.permutation(len(target))
            target = type(target)(
                target.name, "".join(np.array(list(target.sequence))[order]),
                "".join(np.array(list(target.three_di))[order]), target.ca_coord,
                None, None if target.target_codes is None else target.target_codes[order],
            )
        _, score = align(query, target, scheme)
        rows.append(float(score))
        lengths.append((len(query), len(target)))
        per_query.setdefault(identifier_a, profile)
        counts = np.bincount(target.pair_codes(kernel.shape[0] if kernel is not None else len(THREE_DI)),
                             minlength=len(AMINO_ACIDS) * (kernel.shape[0] if kernel is not None else len(THREE_DI)))
        symbol_counts = counts if symbol_counts is None else symbol_counts + counts
        if len(rows) % 200 == 0:
            print(f"  {len(rows)} shuffled pairs scored", flush=True)

    scores = np.array(rows)
    lengths = np.array(lengths)
    area = lengths[:, 0].astype(float) * lengths[:, 1]
    lam, k = fit_lambda_k(scores, area, args.tail_quantile)
    print(f"\n{len(scores)} {args.null} pairs; scores {scores.min():.1f} to {scores.max():.1f}, "
          f"median {np.median(scores):.1f}; fitted above the {args.tail_quantile:.0%} quantile "
          f"({np.quantile(scores, args.tail_quantile):.1f})")
    print(f"fitted:  lambda {lam:.4f}   K {k:.4g}   (bit score = (lambda*S - ln K)/ln 2)")

    # Calibration check: P(S >= x) should be 1 - exp(-E).
    # Calibration: each pair has its own area, so compare per-pair E-values.
    print("\ncalibration on the null (fraction of pairs whose own E-value is at or below x):")
    pair_e = k * area * np.exp(-lam * scores)
    for target_e in (10.0, 1.0, 0.1, 0.01, 0.001):
        print(f"   E <= {target_e:<6} expected {1 - math.exp(-target_e):.4f}   "
              f"observed {float((pair_e <= target_e).mean()):.4f}")

    # Per-query lambda: a profile is its own scoring system.
    background = symbol_counts / symbol_counts.sum()
    n_codes = kernel.shape[0] if kernel is not None else len(THREE_DI)
    amino_scores, _ = _base_scores()
    lambdas = []
    for name, profile in list(per_query.items())[:200]:
        structural = profile_term(profile, args.mode, kernel)
        amino = np.array([AMINO_ACIDS.index(c) if c in AMINO_ACIDS else AMINO_ACIDS.index("X")
                          for c in cache[name].sequence])
        matrix = (args.weight_amino * amino_scores[amino][:, :, None]
                  + args.weight_3di * structural[:, None, :]).reshape(len(amino), -1)
        lambdas.append(analytic_lambda(matrix, background))
    lambdas = np.array([x for x in lambdas if np.isfinite(x)])
    print(f"\nper-query analytic lambda over {len(lambdas)} queries: median {np.median(lambdas):.4f}, "
          f"5-95% {np.percentile(lambdas, 5):.4f}-{np.percentile(lambdas, 95):.4f} "
          f"(spread {np.percentile(lambdas, 95) / np.percentile(lambdas, 5):.2f}x)")
    print(f"   an E-value computed with the median lambda is off by up to "
          f"exp(|dlambda| * S): at S = 100 that is "
          f"{math.exp(abs(np.percentile(lambdas, 95) - np.median(lambdas)) * 100):.1f}x")

    if args.output:
        with open(args.output, "w") as handle:
            json.dump(dict(mode=args.mode, weights=[args.weight_amino, args.weight_3di],
                           gaps=[args.gap_open, args.gap_extend], pairs=len(scores),
                           null=args.null, tail_quantile=args.tail_quantile,
                           lambda_fit=lam, k_fit=k,
                           lambda_per_query_median=float(np.median(lambdas)),
                           lambda_per_query_p5=float(np.percentile(lambdas, 5)),
                           lambda_per_query_p95=float(np.percentile(lambdas, 95)),
                           scores=[float(x) for x in scores],
                           lengths=[[int(a), int(b)] for a, b in lengths]), handle)


if __name__ == "__main__":
    main()
