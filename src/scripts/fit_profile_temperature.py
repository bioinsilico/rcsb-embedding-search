"""Fit a softmax temperature to predicted 3Di profiles, on a held-out split.

A soft 3Di profile is only as good as its probabilities: scored against a target, a
position contributes ``sum_k p(k) * M[k, s]`` (or the log-odds of that mixture), so an
overconfident head behaves like argmax and an underconfident one blurs every position.
Temperature scaling - ``softmax(log p / T)`` - fixes the scale without changing the
argmax, and fitting T on the validation split keeps the benchmark's test pairs out of
the tuning.

Reads ``profiles.npz`` from ``predict_three_di.py --split val`` and the label FASTAs,
minimizes the NLL over the loss-masked residues, and reports NLL, ECE and mean
confidence before and after.

    python fit_profile_temperature.py --profiles predictions/tf/val/profiles.npz \\
        --labels .../three-di/esm3-sequence
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from dataset.three_di_from_embeddings_dataset import STATE_INDEX, read_fasta

N_BINS = 15


def gather(profiles_path: str, labels_path: str, use_reasons: str) -> tuple[np.ndarray, np.ndarray]:
    """Stack log-probabilities and labels of every usable residue."""
    data = np.load(profiles_path)
    three_di = read_fasta(os.path.join(labels_path, "three_di.fasta"))
    exclusion = read_fasta(os.path.join(labels_path, "exclusion.fasta"))
    log_p, label = [], []
    for key in data.files:
        if key.startswith("__"):
            continue
        states, reasons = three_di[key], exclusion[key]
        values = data[key].astype(np.float64)
        if len(states) != len(values):
            raise ValueError(f"{key}: profile length {len(values)} != label length {len(states)}")
        keep = np.array([r in use_reasons for r in reasons])
        log_p.append(values[keep])
        label.append(np.array([STATE_INDEX[s] for s in states])[keep])
    return np.concatenate(log_p), np.concatenate(label)


def tempered(log_p: np.ndarray, temperature: float) -> np.ndarray:
    scaled = log_p / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    return scaled - np.log(np.exp(scaled).sum(axis=1, keepdims=True))


def metrics(log_p: np.ndarray, label: np.ndarray, temperature: float) -> dict:
    log_q = tempered(log_p, temperature)
    q = np.exp(log_q)
    confidence = q.max(axis=1)
    correct = q.argmax(axis=1) == label
    bins = np.minimum((confidence * N_BINS).astype(int), N_BINS - 1)
    ece = 0.0
    for b in range(N_BINS):
        in_bin = bins == b
        if in_bin.any():
            ece += in_bin.mean() * abs(confidence[in_bin].mean() - correct[in_bin].mean())
    return dict(temperature=temperature, nll=float(-log_q[np.arange(len(label)), label].mean()),
                ece=float(ece), confidence=float(confidence.mean()), accuracy=float(correct.mean()))


def fit(log_p: np.ndarray, label: np.ndarray, low: float = 0.2, high: float = 5.0,
        iterations: int = 40) -> float:
    """Golden-section search on log T; the NLL is unimodal in the temperature."""
    ratio = (np.sqrt(5) - 1) / 2
    a, b = np.log(low), np.log(high)
    nll = lambda log_t: metrics(log_p, label, float(np.exp(log_t)))["nll"]
    c, d = b - ratio * (b - a), a + ratio * (b - a)
    fc, fd = nll(c), nll(d)
    for _ in range(iterations):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - ratio * (b - a)
            fc = nll(c)
        else:
            a, c, fc = c, d, fd
            d = a + ratio * (b - a)
            fd = nll(d)
    return float(np.exp((a + b) / 2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profiles", required=True, help="profiles.npz of a held-out split (val)")
    parser.add_argument("--labels", required=True, help="directory with three_di.fasta and exclusion.fasta")
    parser.add_argument("--use-reasons", default=".", help="exclusion characters scored [.]")
    parser.add_argument("--output", default=None, help="write the fitted temperature as JSON")
    args = parser.parse_args()

    log_p, label = gather(args.profiles, args.labels, args.use_reasons)
    before = metrics(log_p, label, 1.0)
    temperature = fit(log_p, label)
    after = metrics(log_p, label, temperature)
    print(f"{len(label):,} residues from {args.profiles}")
    for name, row in (("T=1", before), (f"T={temperature:.3f}", after)):
        print(f"  {name:<9} NLL {row['nll']:.4f}  ECE {row['ece']:.4f}  "
              f"confidence {row['confidence']:.3f}  accuracy {row['accuracy']:.4f}")
    if args.output:
        with open(args.output, "w") as handle:
            json.dump(dict(profiles=os.path.abspath(args.profiles), residues=len(label),
                           before=before, after=after), handle, indent=2)


if __name__ == "__main__":
    main()
