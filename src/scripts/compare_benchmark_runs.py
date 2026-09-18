"""Compare alignment-benchmark runs of several 3Di heads on the same pairs.

Each input is the ``--output`` JSON of ``structure_alignment_benchmark.py benchmark``.
Runs are joined on (query, target), so only pairs every run evaluated are compared,
which is what makes the comparison paired.

For each identity bin it reports the median residue-pair F1 of one scheme per run,
with a 95% bootstrap interval, plus the part of the exact-3Di gain over BLOSUM62 each
run retains.  It then gives paired intervals on the difference between every run and
the reference: resampling pairs keeps the pairing, which is much tighter than
comparing two independent medians and is what decides whether two heads differ.

    python compare_benchmark_runs.py --reference exact \\
        --run exact=cath_exact.json --run cnn=cath_arch_cnn.json --run tf=cath_arch_transformer.json

A run can name its own scheme with ``NAME=JSON@SCHEME``, which compares columns inside one
file - e.g. a soft profile against the argmax of the same profile at the same scale:

    python compare_benchmark_runs.py --reference hard \\
        --run hard=tf_profile.json@3di_aa_sw:hard --run soft=tf_profile.json@3di_aa_sw:soft
"""
from __future__ import annotations

import argparse
import json

import numpy as np

IDENTITY_BINS = (
    ("all", 0.0, 1.01),
    ("<10%", 0.0, 0.10),
    ("10-20%", 0.10, 0.20),
    ("20-30%", 0.20, 0.30),
    (">=30%", 0.30, 1.01),
)


def load(path: str) -> dict[tuple[str, str], dict]:
    with open(path) as handle:
        return {(r["query"], r["target"]): r for r in json.load(handle)}


def bootstrap(values: np.ndarray, statistic, resamples: int, rng: np.random.Generator) -> tuple[float, float]:
    """95% percentile interval of ``statistic`` over resampled rows."""
    if len(values) < 2:
        return float("nan"), float("nan")
    index = rng.integers(0, len(values), size=(resamples, len(values)))
    stats = np.array([statistic(values[i]) for i in index])
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True, metavar="NAME=JSON[@SCHEME]",
                        help="a benchmark JSON; @SCHEME picks that run's column (default --scheme)")
    parser.add_argument("--reference", required=True, help="run name to compare the others against")
    parser.add_argument("--scheme", default="3di_aa_sw")
    parser.add_argument("--baseline-scheme", default="blosum_sw",
                        help="scheme used today, read from the reference run")
    parser.add_argument("--metric", default="f1", choices=["f1", "f1_shift4", "tm"])
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    runs, run_scheme = {}, {}
    for item in args.run:
        name, spec = item.split("=", 1)
        path, _, scheme = spec.partition("@")
        runs[name], run_scheme[name] = path, scheme or args.scheme
    cache = {}
    for path in set(runs.values()):
        cache[path] = load(path)
    data = {name: cache[path] for name, path in runs.items()}
    if args.reference not in data:
        raise SystemExit(f"--reference {args.reference} is not one of {list(data)}")
    keys = sorted(set.intersection(*(set(d) for d in data.values())))
    if not keys:
        raise SystemExit("no pair is shared by every run")
    names = list(runs)
    rng = np.random.default_rng(args.seed)

    identity = np.array([data[args.reference][k]["identity"] for k in keys])
    score = {n: np.array([data[n][k]["schemes"][run_scheme[n]][args.metric] for k in keys]) for n in names}
    baseline = np.array([data[args.reference][k]["schemes"][args.baseline_scheme][args.metric] for k in keys])
    print(f"{len(keys):,} pairs shared by {len(names)} runs; metric {args.metric}, "
          f"schemes {sorted(set(run_scheme.values()))}, "
          f"baseline {args.baseline_scheme}; {args.resamples} bootstrap resamples\n")

    width = max(len(n) for n in names) + 2
    for label, low, high in IDENTITY_BINS:
        rows = (identity >= low) & (identity < high)
        n = int(rows.sum())
        if n == 0:
            continue
        base = float(np.median(baseline[rows]))
        ceiling = float(np.median(score[args.reference][rows]))
        print(f"== identity {label}  (n={n})   {args.baseline_scheme} median {base:.3f}")
        print(f"   {'run':<{width}} {'median':>7}  {'95% CI':>15}  {'% of gain':>9}  "
              f"{'paired median delta':>20}  {'95% CI':>17}  {'paired mean delta':>18}  {'95% CI':>17}")
        for name in names:
            values = score[name][rows]
            median = float(np.median(values))
            lo, hi = bootstrap(values, np.median, args.resamples, rng)
            gain = 100 * (median - base) / (ceiling - base) if ceiling != base else float("nan")
            if name == args.reference:
                delta_text = f"{'(reference: ' + name + ')':>20}"
            else:
                # Both: many pairs align identically under two close schemes, which pins
                # the median difference at 0 while the mean still moves.
                delta = values - score[args.reference][rows]
                d_lo, d_hi = bootstrap(delta, np.median, args.resamples, rng)
                m_lo, m_hi = bootstrap(delta, np.mean, args.resamples, rng)
                delta_text = (f"{float(np.median(delta)):>+20.4f}  [{d_lo:+.4f}, {d_hi:+.4f}]  "
                              f"{float(np.mean(delta)):>+18.4f}  [{m_lo:+.4f}, {m_hi:+.4f}]")
            print(f"   {name:<{width}} {median:>7.3f}  [{lo:.3f}, {hi:.3f}]  {gain:>8.0f}%  {delta_text}")
        print()


if __name__ == "__main__":
    main()
