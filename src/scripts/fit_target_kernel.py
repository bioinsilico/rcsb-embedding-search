"""Fit the target-side kernel: what a predicted 3Di code really means.

A sequence-only database has no structure, so its targets carry *predicted* 3Di.  Storing
a full profile per residue is impractical at UniProt scale (40-80 bytes per residue), and
the alignment's position-specific matrix needs the target to be a small alphabet anyway.
So a target residue is stored as one byte: its predicted state plus a confidence bucket
(20 states x N buckets, 240 codes at N=12, still one byte).

This script estimates, on a held-out split, what each code stands for:

    kernel[code, l] = P(true state = l | predicted state = s, confidence in bucket c)

which is the model's confusion matrix conditioned on confidence.  The aligner then scores
a query profile against a target code by mixing over both distributions at once, so target
uncertainty is handled the same way as query uncertainty.  Codes with no support fall back
to the one-hot predicted state.

    python fit_target_kernel.py --profiles predictions/tf/val/profiles.npz \\
        --labels .../three-di/esm3-sequence --buckets 12 --output target_kernel.npz
"""
from __future__ import annotations

import argparse

import numpy as np

from dataset.three_di_from_embeddings_dataset import STATE_INDEX, read_fasta
from dataset.utils.three_di_labels import THREE_DI
from fit_profile_temperature import gather

N_STATES = len(THREE_DI)


def bucket_of(confidence: np.ndarray, buckets: int) -> np.ndarray:
    """Confidence 1/20..1 mapped to ``buckets`` equal bins of the usable range."""
    lowest = 1.0 / N_STATES
    scaled = (confidence - lowest) / (1.0 - lowest)
    return np.clip((scaled * buckets).astype(int), 0, buckets - 1)


def fit(log_p: np.ndarray, label: np.ndarray, buckets: int, temperature: float,
        prior: float) -> tuple[np.ndarray, np.ndarray]:
    """``(kernel (20*buckets, 20), counts (20*buckets,))`` from held-out residues."""
    scaled = log_p / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    probability = np.exp(scaled)
    probability /= probability.sum(axis=1, keepdims=True)
    predicted = probability.argmax(axis=1)
    code = predicted * buckets + bucket_of(probability.max(axis=1), buckets)

    kernel = np.zeros((N_STATES * buckets, N_STATES))
    np.add.at(kernel, (code, label), 1.0)
    counts = kernel.sum(axis=1)
    # Smooth towards the code's own state, so a rare code cannot produce a wild row and an
    # unseen one reduces to the plain predicted state (the 1-byte argmax representation).
    for state in range(N_STATES):
        kernel[state * buckets:(state + 1) * buckets, state] += prior
    return kernel / kernel.sum(axis=1, keepdims=True), counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profiles", required=True, help="profiles.npz of a held-out split (val)")
    parser.add_argument("--labels", required=True, help="directory with three_di.fasta and exclusion.fasta")
    parser.add_argument("--buckets", type=int, default=12,
                        help="confidence buckets; 20 states x buckets must fit a byte [12]")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--prior", type=float, default=1.0,
                        help="pseudo-count on the code's own state [1.0]")
    parser.add_argument("--use-reasons", default=".")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if N_STATES * args.buckets > 256:
        raise SystemExit(f"{N_STATES} x {args.buckets} codes does not fit in one byte")

    log_p, label = gather(args.profiles, args.labels, args.use_reasons)
    kernel, counts = fit(log_p, label, args.buckets, args.temperature, args.prior)
    np.savez(args.output, kernel=kernel, counts=counts, buckets=np.array(args.buckets),
             alphabet=np.array(list(THREE_DI)))

    print(f"{len(label):,} residues -> {kernel.shape[0]} codes ({args.buckets} buckets per state)")
    print(f"codes with no support: {(counts == 0).sum()}; median support {int(np.median(counts)):,}")
    # How sharp a code is: the probability it assigns to its own predicted state.
    own = np.array([kernel[s * args.buckets + b, s] for s in range(N_STATES) for b in range(args.buckets)])
    for b in range(args.buckets):
        rows = [s * args.buckets + b for s in range(N_STATES)]
        support = counts[rows].sum()
        if support:
            print(f"  bucket {b:>2}: {support:>9,.0f} residues, mean P(true = predicted) "
                  f"{own[rows].mean():.3f}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
