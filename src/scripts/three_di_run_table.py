"""Compare 3Di head runs: one row per TensorBoard run directory.

    python three_di_run_table.py --runs /pscratch/sd/j/jsegura/three-di-runs

Sorted by ``expected_score`` (mean ``sum_k p(k) * M3Di[k, true]``) rather than by
accuracy, because that is the quantity the alignment scorer consumes and because two
heads with the same accuracy can differ in how costly their mistakes are.  The final
column is the share of the ceiling a perfect prediction would reach on the same labels.

Per-residue accuracy is reported but is not the decision metric: rank the top few runs
with ``predict_three_di.py`` plus ``structure_alignment_benchmark.py --query-3di``.
"""
from __future__ import annotations

import argparse
import glob
import os

METRICS = ("accuracy", "top3_accuracy", "expected_score", "expected_score_ceiling",
           "nll", "ece", "confidence", "validation_loss")


def read_run(directory: str) -> dict | None:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    accumulator = EventAccumulator(directory, size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = accumulator.Tags()["scalars"]
    if "expected_score" not in tags:
        return None
    row = {"run": os.path.basename(os.path.dirname(directory)), "version": os.path.basename(directory)}
    for metric in METRICS:
        if metric in tags:
            series = accumulator.Scalars(metric)
            row[metric] = series[-1].value
            if metric == "expected_score":
                row["best_expected_score"] = max(s.value for s in series)
                row["epochs"] = len(series)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", required=True, help="directory holding <name>/version_N run folders")
    parser.add_argument("--sort", default="best_expected_score")
    args = parser.parse_args()

    rows = [row for directory in sorted(glob.glob(os.path.join(args.runs, "*", "version_*")))
            if (row := read_run(directory)) is not None]
    if not rows:
        raise SystemExit(f"no runs with metrics under {args.runs}")
    rows.sort(key=lambda r: -r.get(args.sort, float("-inf")))

    header = (f"{'run':<18} {'ver':<10} {'ep':>3} {'accuracy':>9} {'top3':>7} {'exp.score':>10} "
              f"{'best':>7} {'ceiling':>8} {'% ceil':>7} {'nll':>6} {'ece':>6} {'conf':>6}")
    print(header)
    print("-" * len(header))
    for row in rows:
        ceiling = row.get("expected_score_ceiling", float("nan"))
        share = 100 * row.get("best_expected_score", float("nan")) / ceiling if ceiling else float("nan")
        print(f"{row['run']:<18} {row['version']:<10} {row.get('epochs', 0):>3} "
              f"{row.get('accuracy', float('nan')):>9.4f} {row.get('top3_accuracy', float('nan')):>7.4f} "
              f"{row.get('expected_score', float('nan')):>10.4f} "
              f"{row.get('best_expected_score', float('nan')):>7.4f} {ceiling:>8.4f} {share:>6.1f}% "
              f"{row.get('nll', float('nan')):>6.3f} {row.get('ece', float('nan')):>6.3f} "
              f"{row.get('confidence', float('nan')):>6.3f}")


if __name__ == "__main__":
    main()
