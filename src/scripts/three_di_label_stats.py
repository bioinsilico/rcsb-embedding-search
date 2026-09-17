"""Inspect the 3Di training labels ``parse_domain`` produces for a set of domain PDBs.

Reports how many files are unusable and why, how much of each domain the loss
mask removes (broken down by exclusion reason), the class balance of usable
labels, and prints a few domains as aligned sequence / 3Di / exclusion lines.

``--check`` also verifies, per domain, that the states agree with
``biotite.structure.alphabet.to_3di`` wherever the encoder produced a state, i.e.
that re-implementing the encoder steps to keep its mask changed nothing else.

    python three_di_label_stats.py --pdb-dir /data/cath_23M/pdb --limit 2000 --check
"""
from __future__ import annotations

import argparse
import os
import random
import warnings
from collections import Counter
from multiprocessing import Pool

import numpy as np
from biotite.structure.alphabet import to_3di

from dataset.utils.three_di_labels import (
    BACKBONE, BREAK, EXCLUSION_REASONS, TERMINUS, THREE_DI,
    UnusableDomain, parse_domain, read_domain_atoms,
)

REASON_SYMBOL = {TERMINUS: "t", BACKBONE: "b", BREAK: "x"}


def _analyse(job: tuple[str, bool]) -> dict:
    path, check = job
    warnings.filterwarnings("ignore")
    try:
        domain = parse_domain(path, name=os.path.basename(path))
    except UnusableDomain as error:
        return {"path": path, "error": str(error).split(": ", 1)[-1]}
    except Exception as error:  # surface anything unexpected instead of dying in a worker
        return {"path": path, "error": f"UNEXPECTED {type(error).__name__}: {error}"}
    result = {"path": path, "domain": domain}
    if check:
        reference = str(to_3di(read_domain_atoms(path))[0][0])
        ours = "".join(THREE_DI[s] for s in domain.three_di)
        encoded = (domain.exclusion & (TERMINUS | BACKBONE)) == 0
        result["check_mismatch"] = sum(
            a != b for a, b, e in zip(ours, reference, encoded) if e
        )
        result["check_length"] = len(reference) == len(domain)
    return result


def _error_category(message: str) -> str:
    if message.startswith("UNEXPECTED"):
        return message
    if "0 models" in message:
        return "empty file"
    if "disjoint" in message:
        return "disjoint segments"
    if "chain ids" in message:
        return "multiple chain ids"
    if "only" in message:
        return "too short"
    return message


def _show(domain, width: int = 100) -> None:
    exclusion = "".join(
        "." if e == 0 else next(REASON_SYMBOL[bit] for bit in (BREAK, BACKBONE, TERMINUS) if e & bit)
        for e in domain.exclusion
    )
    print(f"\n{domain.name}  L={len(domain)}  usable={domain.mask.mean():.1%}")
    for start in range(0, len(domain), width):
        stop = start + width
        print(f"  {start + 1:>5} seq  {domain.sequence[start:stop]}")
        print(f"        3di  {domain.three_di_string[start:stop]}")
        print(f"        mask {exclusion[start:stop]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdb-dir", required=True, nargs="+")
    parser.add_argument("--limit", type=int, default=None, help="random sample size per directory")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--show", type=int, default=3, help="domains to print residue by residue")
    parser.add_argument("--check", action="store_true", help="compare states with biotite to_3di")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    paths = []
    for directory in args.pdb_dir:
        files = sorted(f for f in os.listdir(directory) if f.endswith(".pdb"))
        if args.limit is not None and args.limit < len(files):
            files = rng.sample(files, args.limit)
        paths.extend(os.path.join(directory, f) for f in files)

    with Pool(args.workers) as pool:
        results = pool.map(_analyse, [(p, args.check) for p in paths], chunksize=16)

    errors = Counter(_error_category(r["error"]) for r in results if "error" in r)
    domains = [r["domain"] for r in results if "domain" in r]
    print(f"files    : {len(results):,}  usable {len(domains):,}  unusable {sum(errors.values()):,}")
    for category, count in errors.most_common():
        print(f"           {count:>6,}  {category}")

    exclusion = np.concatenate([d.exclusion for d in domains])
    lengths = np.array([len(d) for d in domains])
    usable = np.array([d.mask.mean() for d in domains])
    print(f"\nresidues : {len(exclusion):,}  (domain length median {int(np.median(lengths))}, "
          f"range {lengths.min()}-{lengths.max()})")
    print(f"usable   : {(exclusion == 0).mean():.2%} of residues")
    for bit, reason in EXCLUSION_REASONS.items():
        print(f"excluded : {((exclusion & bit) != 0).mean():6.2%}  {reason} (reasons can overlap)")
    print(f"per domain usable share: median {np.median(usable):.1%}, p10 {np.percentile(usable, 10):.1%}, "
          f"domains < 50% usable {(usable < 0.5).sum():,}")

    states = np.concatenate([d.three_di[d.mask] for d in domains])
    counts = np.bincount(states, minlength=len(THREE_DI)) / len(states)
    masked_d = np.mean(np.concatenate([d.three_di[~d.mask] for d in domains]) == THREE_DI.index("d"))
    print("\nusable label distribution:")
    print("  " + "  ".join(f"{s}:{p:.1%}" for s, p in sorted(zip(THREE_DI, counts), key=lambda x: -x[1])))
    print(f"share of 'd' among usable labels {counts[THREE_DI.index('d')]:.1%}; "
          f"among excluded residues (encoder fill value) {masked_d:.1%}")

    if args.check:
        checked = [r for r in results if "check_mismatch" in r]
        bad = [r for r in checked if r["check_mismatch"] or not r["check_length"]]
        print(f"\nto_3di check: {len(checked):,} domains, {len(bad):,} disagree")
        for r in bad[:5]:
            print(f"  {r['path']}: {r['check_mismatch']} state mismatches, length ok {r['check_length']}")

    for domain in rng.sample(domains, min(args.show, len(domains))):
        _show(domain)


if __name__ == "__main__":
    main()
