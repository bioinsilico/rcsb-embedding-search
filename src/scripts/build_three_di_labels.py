"""Precompute the 3Di training labels of a split manifest into three FASTA files.

Computing labels inside the Dataset costs a small-file read per sample.  On the Mac
that is 5 ms; on NERSC's shared filesystem it is 41 ms median with a 750 ms tail, so
one worker sustains ~21 domains/s and a cached-embedding head would sit idle waiting
for labels - and one random small-file read per __getitem__, times ranks times
workers, is the metadata storm that has already cost this project a 64-node job.
The labels are 8 MB in total, so precomputing lets the Dataset hold all of them in
memory and do no I/O at all.

Three FASTA files are written, sharing ids and order:

* ``sequences.fasta``  - amino acids, exactly what was fed to ESM3
* ``three_di.fasta``   - 3Di state per residue (all of them, including excluded ones)
* ``exclusion.fasta``  - why each residue is excluded: ``.`` usable, ``t`` terminus,
  ``b`` backbone, ``x`` chain break (see ``dataset.utils.three_di_labels``)

The exclusion file stores the *reason* rather than a boolean mask so that the masking
policy stays a training-time decision: keeping break-adjacent residues, or weighting
them down instead of dropping them, needs no re-parsing of 47k structures.

    python build_three_di_labels.py --manifest domain_split.tsv --out-dir esm3-sequence
"""
from __future__ import annotations

import argparse
import os
import warnings
from collections import Counter
from multiprocessing import Pool

import numpy as np

from dataset.utils.three_di_labels import (
    BACKBONE, BREAK, TERMINUS, THREE_DI, UnusableDomain, parse_domain,
)

#: One character per exclusion reason; '.' means the residue is usable.
REASON_CHARS = ((BREAK, "x"), (BACKBONE, "b"), (TERMINUS, "t"))
USABLE_CHAR = "."

FILES = ("sequences.fasta", "three_di.fasta", "exclusion.fasta")


def exclusion_string(exclusion: np.ndarray) -> str:
    """Per-residue reason characters; the first matching reason wins."""
    return "".join(
        next((char for bit, char in REASON_CHARS if value & bit), USABLE_CHAR)
        for value in exclusion
    )


def _labels(row: dict) -> dict:
    warnings.filterwarnings("ignore")
    try:
        domain = parse_domain(row["path"], name=row["domain_id"])
    except UnusableDomain as error:
        return {**row, "error": str(error).split(": ", 1)[-1]}
    except Exception as error:
        return {**row, "error": f"{type(error).__name__}: {error}"}
    return {
        **row,
        "sequence": domain.sequence,
        "three_di": "".join(THREE_DI[state] for state in domain.three_di),
        "exclusion": exclusion_string(domain.exclusion),
        "usable": int(domain.mask.sum()),
    }


def read_manifest(path: str) -> list[dict]:
    with open(path) as handle:
        header = handle.readline().rstrip("\n").split("\t")
        return [dict(zip(header, line.rstrip("\n").split("\t"))) for line in handle if line.strip()]


def wrap(sequence: str, width: int) -> str:
    if width <= 0:
        return sequence
    return "\n".join(sequence[i:i + width] for i in range(0, len(sequence), width))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, help="domain_split.tsv from domain_family_split.py")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--line-width", type=int, default=0, help="wrap sequences (0 = one line each)")
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"{len(rows):,} domains from {args.manifest}")

    handles = {name: open(os.path.join(args.out_dir, name + ".tmp"), "w") for name in FILES}
    unusable = open(os.path.join(args.out_dir, "labels_unusable.tsv"), "w")
    unusable.write("domain_id\tsplit\treason\n")

    n_unusable = 0
    lengths, usable_shares = [], []
    reasons = Counter()
    states = Counter()
    by_split = Counter()
    with Pool(args.workers) as pool:
        for result in pool.imap(_labels, rows, chunksize=32):
            if "error" in result:
                unusable.write(f"{result['domain_id']}\t{result['split']}\t{result['error']}\n")
                n_unusable += 1
                continue
            header = (f">{result['domain_id']} source={result['source']} split={result['split']} "
                      f"length={len(result['sequence'])} usable={result['usable']}")
            for name, field in zip(FILES, ("sequence", "three_di", "exclusion")):
                handles[name].write(f"{header}\n{wrap(result[field], args.line_width)}\n")
            lengths.append(len(result["sequence"]))
            usable_shares.append(result["usable"] / max(1, len(result["sequence"])))
            reasons.update(result["exclusion"])
            states.update(s for s, e in zip(result["three_di"], result["exclusion"]) if e == USABLE_CHAR)
            by_split[result["split"]] += 1

    for name, handle in handles.items():
        handle.close()
        os.replace(os.path.join(args.out_dir, name + ".tmp"), os.path.join(args.out_dir, name))
    unusable.close()

    residues = sum(lengths)
    print(f"written  : {len(lengths):,} domains ({dict(by_split)}), {residues:,} residues; "
          f"{n_unusable:,} unusable")
    print(f"usable   : {reasons[USABLE_CHAR] / residues:.2%} of residues; per domain median "
          f"{np.median(usable_shares):.1%}, p10 {np.percentile(usable_shares, 10):.1%}")
    for bit, char in REASON_CHARS:
        name = {BREAK: "break", BACKBONE: "backbone", TERMINUS: "terminus"}[bit]
        print(f"excluded : {reasons[char] / residues:6.2%}  {name} (first matching reason per residue)")
    total = sum(states.values())
    print("label distribution (usable residues):")
    print("  " + "  ".join(f"{s}:{c / total:.1%}" for s, c in states.most_common()))
    print(f"\nwrote {', '.join(os.path.join(args.out_dir, f) for f in FILES)}")


if __name__ == "__main__":
    main()
