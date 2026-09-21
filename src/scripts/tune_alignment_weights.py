"""Tune the 3Di weight and gap penalties of the stage-2 scoring, on validation pairs.

Foldseek's 1.4 / 2.1 weights and 10/1 gaps were set for *hard* 3Di letters on both sides.
A soft query profile changes both the spread of the 3Di term (averaging over states makes
it vary less) and its zero point (in log-odds form an uninformative position scores 0
rather than clearly negative), so the balance against the amino-acid term and against the
cost of a gap is no longer the one those numbers were chosen for.

Only three numbers matter: multiplying every score by a constant and scaling the gaps with
it leaves the alignment unchanged, so the amino-acid weight is fixed at 1.4 and the grid
runs over the 3Di weight and the two gap penalties.

Nothing here touches the test pairs: tune on validation queries, then confirm once on test
with ``structure_alignment_benchmark.py``.

    python tune_alignment_weights.py --pairs cath_val_pairs.csv --structures /data/cath_23M/pdb \\
        --usalign ./USalign --query-profile val/profiles.npz --mode logodds --limit 600
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np

from structure_alignment import (
    AMINO_ACIDS, SCHEMES, THREE_DI, UnreadableStructure, align, global_identity, load_chain,
    prf, reference_alignment,
)
from structure_alignment_benchmark import read_profiles, read_target_codes, sample_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--structures", required=True)
    parser.add_argument("--suffix", default=".pdb")
    parser.add_argument("--usalign", required=True)
    parser.add_argument("--query-profile", required=True)
    parser.add_argument("--target-profile", default=None, help="predicted targets (sequence-only database)")
    parser.add_argument("--target-kernel", default=None)
    parser.add_argument("--profile-temperature", type=float, default=1.0)
    parser.add_argument("--profile-scale", type=int, default=100)
    parser.add_argument("--mode", default="logodds", choices=["argmax", "expected", "logodds"])
    parser.add_argument("--base-scheme", default="3di_aa_sw")
    parser.add_argument("--weights", type=float, nargs="+", default=[1.0, 1.4, 1.7, 2.1, 2.5],
                        help="3Di weights to try (the amino-acid weight stays 1.4)")
    parser.add_argument("--gap-open", type=float, nargs="+", default=[10, 15, 20, 25])
    parser.add_argument("--gap-extend", type=float, nargs="+", default=[1, 2])
    parser.add_argument("--limit", type=int, default=600)
    parser.add_argument("--min-length", type=int, default=60)
    parser.add_argument("--max-length", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--output", default=None)
    parser.add_argument("--dump-per-pair", default=None,
                        help="also write each pair's F1 for every grid point, for paired tests")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    candidates = sample_pairs(args.pairs, 0.7, args.limit * 4, rng)
    profiles = read_profiles(args.query_profile, args.profile_temperature)
    targets = (read_target_codes(args.target_profile, args.target_kernel, args.profile_temperature)
               if args.target_profile else None)
    kernel = np.load(args.target_kernel)["kernel"] if args.target_kernel else None

    # One pass over the pairs: load the structures, run US-align once, keep what the grid needs.
    cache, records, start = {}, [], time.time()
    for identifier_a, identifier_b, _ in candidates:
        if len(records) >= args.limit:
            break
        if identifier_a not in profiles or (targets is not None and identifier_b not in targets):
            continue
        path_a = os.path.join(args.structures, identifier_a + args.suffix)
        path_b = os.path.join(args.structures, identifier_b + args.suffix)
        if not (os.path.exists(path_a) and os.path.exists(path_b)):
            continue
        try:
            for identifier, path in ((identifier_a, path_a), (identifier_b, path_b)):
                if identifier not in cache:
                    cache[identifier] = load_chain(path, name=identifier)
            query, target = cache[identifier_a], cache[identifier_b]
        except (UnreadableStructure, ValueError, KeyError):
            continue
        if not (args.min_length <= len(query) <= args.max_length
                and args.min_length <= len(target) <= args.max_length):
            continue
        profile = profiles[identifier_a]
        if len(profile) != len(query):
            continue
        if targets is not None:
            string, codes = targets[identifier_b]
            if len(string) != len(target):
                continue
            target = type(target)(target.name, target.sequence, string, target.ca_coord,
                                  None, codes)
        reference = reference_alignment(args.usalign, path_a, path_b)
        if reference is None or reference.get("pairs") is None or len(reference["pairs"]) == 0:
            continue
        query = type(query)(query.name, query.sequence, query.three_di, query.ca_coord, profile, None)
        records.append(dict(query=query, target=target, reference=reference["pairs"],
                            identity=global_identity(query, target)))
        if len(records) % 100 == 0:
            print(f"  {len(records)} pairs prepared in {time.time() - start:.0f}s", flush=True)
    if not records:
        raise SystemExit("no usable pairs")
    identity = np.array([r["identity"] for r in records])
    print(f"{len(records)} validation pairs ({int((identity < 0.2).sum())} below 20% identity); "
          f"mode {args.mode}, targets {'predicted' if targets is not None else 'exact'}")

    results = []
    for weight in args.weights:
        for gap_open in args.gap_open:
            for gap_extend in args.gap_extend:
                scheme = dict(SCHEMES[args.base_scheme], weights=(1.4, weight),
                              gaps=(-gap_open, -gap_extend), profile=args.mode,
                              scale=args.profile_scale, target_kernel=kernel)
                f1 = np.array([
                    prf(align(r["query"], r["target"], scheme)[0], r["reference"])[2]
                    for r in records
                ])
                row = dict(weight_3di=weight, gap_open=gap_open, gap_extend=gap_extend,
                           f1=float(f1.mean()), f1_low_identity=float(f1[identity < 0.2].mean()))
                row["per_pair"] = [float(x) for x in f1] if args.dump_per_pair else None
                results.append(row)
                print(f"   w3di {weight:>4} gaps {gap_open:>4.0f}/{gap_extend:<3.0f} "
                      f"F1 {row['f1']:.4f}  (<20% id {row['f1_low_identity']:.4f})", flush=True)

    results.sort(key=lambda r: -r["f1"])
    best = results[0]
    default = next((r for r in results if r["weight_3di"] == 2.1 and r["gap_open"] == 10
                    and r["gap_extend"] == 1), None)
    print(f"\nbest: 3Di weight {best['weight_3di']}, gaps {best['gap_open']:.0f}/{best['gap_extend']:.0f} "
          f"-> F1 {best['f1']:.4f}")
    if default:
        print(f"Foldseek default (2.1, 10/1)                    -> F1 {default['f1']:.4f} "
              f"({best['f1'] - default['f1']:+.4f})")
    payload = dict(pairs=len(records), mode=args.mode,
                   targets="predicted" if targets is not None else "exact",
                   queries=[r["query"].name for r in records],
                   identity=[float(i) for i in identity],
                   results=[{k: v for k, v in r.items() if k != "per_pair"} for r in results])
    if args.output:
        with open(args.output, "w") as handle:
            json.dump(payload, handle, indent=2)
    if args.dump_per_pair:
        with open(args.dump_per_pair, "w") as handle:
            json.dump(dict(payload, results=results,
                           targets_list=[r["target"].name for r in records]), handle)


if __name__ == "__main__":
    main()
