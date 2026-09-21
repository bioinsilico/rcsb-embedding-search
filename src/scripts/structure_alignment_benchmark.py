"""Measure how close sequence-level alignments come to structural alignments.

Three sub-commands:

``pair``       align two chains under every scheme and report both quality measures.
``benchmark``  run over a table of structure pairs with known TM-scores (CATH, SCOP).
``sweep``      degrade the query's 3Di to a target accuracy and re-measure, which estimates
               what a predicted (rather than exact) 3Di string would cost.

Quality is measured two ways: the TM-score of the superposition each alignment implies
(needs no reference), and precision / recall / F1 of residue pairs against a US-align
reference alignment. Results are stratified by sequence identity, since that is where the
schemes diverge.

US-align is required for the reference measures. Build it with::

    git clone --depth 1 https://github.com/pylelab/USalign.git && cd USalign && make
    # on macOS, if the C++ headers are not on the default search path:
    #   SDK=$(xcrun --show-sdk-path)
    #   clang++ -O3 -ffast-math -isysroot $SDK -cxx-isystem $SDK/usr/include/c++/v1 \
    #       -o USalign USalign.cpp

Examples::

    python src/scripts/structure_alignment_benchmark.py pair \
        --query /data/pdb/3ZI1.cif.gz --query-chain A \
        --target /data/pdb/1ZSW.cif --target-chain A --usalign ./USalign

    python src/scripts/structure_alignment_benchmark.py benchmark \
        --pairs /data/cath_23M/cath_23M.csv --structures /data/cath_23M/pdb \
        --limit 400 --usalign ./USalign --output cath_benchmark.json

    python src/scripts/structure_alignment_benchmark.py sweep \
        --pairs /data/scop-zenodo/TMfast.dual.csv --structures /data/scop-zenodo/pdb \
        --limit 200 --usalign ./USalign

    # Predicted 3Di profiles (predict_three_di.py profiles.npz): adds, per --profile-schemes
    # base, one column per --profile-mode and --min-confidence, e.g. 3di_aa_sw:soft>0.5.
    python src/scripts/structure_alignment_benchmark.py benchmark \
        --pairs test_pairs.csv --structures /data/cath_23M/pdb --usalign ./USalign \
        --query-profile predictions/transformer/profiles.npz \
        --profile-mode argmax expected logodds --min-confidence 0.5
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics as st
import sys
import tempfile
import time
from dataclasses import replace

import numpy as np

from structure_alignment import (
    SCHEMES,
    THREE_DI,
    UnreadableStructure,
    align,
    degrade_3di,
    global_identity,
    load_chain,
    prf,
    reference_alignment,
    reference_tm,
    sequence_identity,
    tm_score,
    write_ca_pdb,
)

IDENTITY_BINS = [
    ("all", lambda value: True),
    ("id <10%", lambda value: value < 0.10),
    ("10-20%", lambda value: 0.10 <= value < 0.20),
    ("20-30%", lambda value: 0.20 <= value < 0.30),
    (">=30%", lambda value: value >= 0.30),
]


def median(values):
    values = [v for v in values if v is not None]
    return st.median(values) if values else float("nan")


def print_table(title, rows, columns, cell, note=None):
    width = max([13] + [len(c) + 1 for c in columns])
    print(f"\n{title}")
    print(f"{'subset':<10}{'n':>5}  " + "".join(f"{c:>{width}}" for c in columns))
    for label, selected in rows:
        if not selected:
            continue
        print(f"{label:<10}{len(selected):>5}  " + "".join(f"{cell(selected, c):>{width}.3f}" for c in columns))
    if note:
        print(f"  ({note})")


def sample_pairs(path, min_tm, limit, rng, line_rate=1.0):
    """Reservoir-sample pairs whose TM-score passes ``min_tm``.

    Accepts ``id1,id2,tm`` and ``id1,id2,tm1,tm2`` tables (the larger TM is used), so both
    the CATH pair list and the SCOP ``TMfast.dual.csv`` work. ``line_rate`` below 1 skips
    lines while reading, which keeps very large tables cheap.
    """
    reservoir, seen = [], 0
    with open(path) as handle:
        for line in handle:
            if line_rate < 1.0 and rng.random() > line_rate:
                continue
            fields = line.strip().split(",")
            if len(fields) < 3:
                continue
            try:
                tm = max(float(v) for v in fields[2:])
            except ValueError:
                continue
            if tm < min_tm:
                continue
            seen += 1
            if len(reservoir) < limit:
                reservoir.append((fields[0], fields[1], tm))
            else:
                index = rng.randrange(seen)
                if index < limit:
                    reservoir[index] = (fields[0], fields[1], tm)
    rng.shuffle(reservoir)
    return reservoir


def evaluate_pair(query, target, reference, schemes=SCHEMES):
    """Every scheme's alignment quality for one pair (profile schemes need a query profile)."""
    length_norm = min(len(query), len(target))
    out = {}
    for name, scheme in schemes.items():
        if scheme.get("profile") and query.three_di_profile is None:
            continue
        pairs, score = align(query, target, scheme)
        entry = dict(
            tm=tm_score(pairs, query, target, length_norm),
            n_aligned=len(pairs),
            identity=sequence_identity(pairs, query, target),
            coverage=len(pairs) / length_norm,
            score=score,
        )
        if reference is not None:
            precision, recall, f1 = prf(pairs, reference["pairs"], 0)
            entry.update(precision=precision, recall=recall, f1=f1,
                         f1_shift4=prf(pairs, reference["pairs"], 4)[2])
        out[name] = entry
    return out


def usable_reference(usalign, path_query, path_target, query, target):
    """US-align reference, or None when it did not parse the same residues we did."""
    if not usalign:
        return None
    reference = reference_alignment(usalign, path_query, path_target)
    if reference is None:
        return None
    if reference["sequence_query"] != query.sequence or reference["sequence_target"] != target.sequence:
        return None
    return reference


def command_pair(args):
    query = load_chain(args.query, args.query_chain, name=os.path.basename(args.query))
    target = load_chain(args.target, args.target_chain, name=os.path.basename(args.target))
    print(f"{query.name}: {len(query)} residues, 3Di unencodable {query.invalid_3di_fraction:.1%}")
    print(f"{target.name}: {len(target)} residues, 3Di unencodable {target.invalid_3di_fraction:.1%}")
    print(f"global sequence identity: {global_identity(query, target):.1%}")

    reference = None
    if args.usalign:
        # Write CA-only copies so US-align sees exactly the residues loaded here.
        with tempfile.TemporaryDirectory() as tmp:
            query_pdb, target_pdb = f"{tmp}/query.pdb", f"{tmp}/target.pdb"
            write_ca_pdb(query, query_pdb)
            write_ca_pdb(target, target_pdb)
            reference = usable_reference(args.usalign, query_pdb, target_pdb, query, target)
            results = evaluate_pair(query, target, reference)
        if reference is not None:
            length_norm = min(len(query), len(target))
            print(f"US-align reference: TM={reference_tm(reference, length_norm):.3f}, "
                  f"{len(reference['pairs'])} aligned pairs")
        else:
            print("US-align reference unavailable (residue sets differ); F1 not reported")
    else:
        results = evaluate_pair(query, target, None)

    print()
    header = f"{'scheme':<13}{'TM':>7}{'aligned':>9}{'identity':>10}{'coverage':>10}"
    if reference is not None:
        header += f"{'F1':>8}{'F1 +/-4':>9}"
    print(header)
    for name, entry in results.items():
        line = (f"{name:<13}{entry['tm']:>7.3f}{entry['n_aligned']:>9}"
                f"{entry['identity']:>10.2f}{entry['coverage']:>10.2f}")
        if reference is not None:
            line += f"{entry['f1']:>8.3f}{entry['f1_shift4']:>9.3f}"
        print(line)


def read_fasta(path):
    """``{id: sequence}``; the id is the first token of the header."""
    records, name, chunks = {}, None, []
    with open(path) as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if name is not None:
                    records[name] = "".join(chunks)
                name, chunks = line[1:].split()[0], []
            elif line:
                chunks.append(line)
    if name is not None:
        records[name] = "".join(chunks)
    return records


PROFILE_TAGS = {"argmax": "hard", "expected": "soft", "logodds": "lo"}


def read_target_codes(path, kernel_path, temperature=1.0):
    """``{id: (argmax 3Di string, codes)}`` for targets whose 3Di is predicted.

    A sequence-only database has no structure, so its targets carry predicted 3Di. Each
    residue becomes one byte - the predicted state plus a confidence bucket - and the kernel
    (fit_target_kernel.py) says what that code implies. Without a kernel the target is just
    the predicted 3Di string, which is the plain 1-byte representation.
    """
    profiles = read_profiles(path, temperature)
    buckets = int(np.load(kernel_path)["buckets"]) if kernel_path else None
    out = {}
    for name, p in profiles.items():
        state = p.argmax(axis=1)
        string = "".join(THREE_DI[k] for k in state)
        if buckets is None:
            out[name] = (string, None)
            continue
        lowest = 1.0 / len(THREE_DI)
        scaled = (p.max(axis=1) - lowest) / (1.0 - lowest)
        bucket = np.clip((scaled * buckets).astype(int), 0, buckets - 1)
        out[name] = (string, state * buckets + bucket)
    return out


def read_profiles(path, temperature=1.0):
    """``{id: (L, 20) probabilities}`` from an .npz of log-probabilities.

    ``temperature`` rescales the log-probabilities before normalizing, which equals
    softmax(logits / T) because a log-softmax differs from the logits by a constant.
    """
    data = np.load(path)
    if "__alphabet__" in data.files and "".join(data["__alphabet__"]) != THREE_DI:
        sys.exit(f"{path}: 3Di state order differs from {THREE_DI}")
    profiles = {}
    for key in data.files:
        if key.startswith("__"):
            continue
        log_p = data[key].astype(np.float64) / temperature
        p = np.exp(log_p - log_p.max(axis=1, keepdims=True))
        profiles[key] = p / p.sum(axis=1, keepdims=True)
    return profiles


def profile_schemes(args):
    """Extra schemes built from the profile flags, e.g. ``3di_aa_sw:soft>0.5``.

    With ``--target-kernel`` each one also gets a ``+tk`` variant, which reads the target's
    confidence code through the kernel instead of taking its predicted state at face value.
    """
    if not getattr(args, "query_profile", None):
        return {}
    kernel = np.load(args.target_kernel)["kernel"] if getattr(args, "target_kernel", None) else None
    out = {}
    for base in args.profile_schemes:
        for mode in args.profile_mode:
            for threshold in [None] + list(args.min_confidence):
                name = f"{base}:{PROFILE_TAGS[mode]}" + ("" if threshold is None else f">{threshold:g}")
                out[name] = dict(SCHEMES[base], profile=mode, scale=args.profile_scale,
                                 min_confidence=threshold)
                if kernel is not None:
                    out[name + "+tk"] = dict(out[name], target_kernel=kernel)
    return out


def collect(args, need_reference):
    """Load, align and score sampled pairs; returns one record per usable pair."""
    rng = random.Random(args.seed)
    candidates = sample_pairs(args.pairs, args.min_tm, args.limit * args.oversample, rng,
                              args.line_rate)
    print(f"{len(candidates)} candidate pairs with TM >= {args.min_tm}", flush=True)
    cache, records, start = {}, [], time.time()
    # Predicted query 3Di, if given: only pairs whose query has a prediction are kept,
    # which is also how the evaluation stays inside the held-out split.
    predicted = read_fasta(args.query_3di) if getattr(args, "query_3di", None) else None
    if getattr(args, "query_profile", None):
        if predicted is not None:
            sys.exit("--query-3di and --query-profile are exclusive: the profile's argmax is the string")
        profiles = read_profiles(args.query_profile, args.profile_temperature)
        # The argmax string comes from the same file, so string and profile schemes agree.
        predicted = {name: "".join(THREE_DI[k] for k in p.argmax(axis=1))
                     for name, p in profiles.items()}
    else:
        profiles = None
    # Predicted targets, for a sequence-only database.
    targets = (read_target_codes(args.target_profile, getattr(args, "target_kernel", None),
                                 args.profile_temperature)
               if getattr(args, "target_profile", None) else None)
    skipped_target = 0
    skipped_length = 0

    def chain(identifier):
        if identifier not in cache:
            if len(cache) > 2000:
                cache.clear()
            cache[identifier] = load_chain(
                os.path.join(args.structures, identifier + args.suffix), name=identifier)
        return cache[identifier]

    for identifier_a, identifier_b, tm in candidates:
        if len(records) >= args.limit:
            break
        path_a = os.path.join(args.structures, identifier_a + args.suffix)
        path_b = os.path.join(args.structures, identifier_b + args.suffix)
        if not (os.path.exists(path_a) and os.path.exists(path_b)):
            continue
        if predicted is not None and identifier_a not in predicted:
            continue
        try:
            query, target = chain(identifier_a), chain(identifier_b)
        except (UnreadableStructure, ValueError, KeyError):
            continue
        if predicted is not None:
            # The prediction is indexed by the label parser's residues, which keeps
            # residues without a CA atom; load_chain keeps only CA-bearing ones. When the
            # two disagree the strings cannot be aligned index for index, so skip the pair.
            if len(predicted[identifier_a]) != len(query):
                skipped_length += 1
                continue
            query = replace(query, three_di=predicted[identifier_a],
                            three_di_profile=None if profiles is None else profiles[identifier_a])
        if targets is not None:
            # Both sides must be predicted for this to measure a sequence-only database.
            if identifier_b not in targets:
                skipped_target += 1
                continue
            string, codes = targets[identifier_b]
            if len(string) != len(target):
                skipped_length += 1
                continue
            target = replace(target, three_di=string, target_codes=codes)
        if not (args.min_length <= len(query) <= args.max_length
                and args.min_length <= len(target) <= args.max_length):
            continue
        reference = usable_reference(args.usalign, path_a, path_b, query, target)
        if need_reference and reference is None:
            continue
        records.append(dict(query=identifier_a, target=identifier_b, reference_tm_table=tm,
                            identity=global_identity(query, target),
                            reference_tm=reference_tm(reference, min(len(query), len(target)))
                            if reference else None,
                            chains=(query, target), reference=reference))
        if len(records) % 50 == 0:
            print(f"  {len(records)} pairs in {time.time() - start:.0f}s", flush=True)
    if targets is not None:
        print(f"predicted target 3Di from {args.target_profile}"
              + (f" with kernel {args.target_kernel}" if getattr(args, "target_kernel", None) else
                 " (predicted state only, no confidence)")
              + f": {skipped_target} pairs skipped for a target without a prediction")
    if predicted is not None:
        print(f"predicted query 3Di from {args.query_3di or args.query_profile}: "
              f"{len(predicted):,} domains available, "
              f"{skipped_length} pairs skipped on a length mismatch with the structure")
    return records


def command_benchmark(args):
    records = collect(args, need_reference=False)
    schemes = {**SCHEMES, **profile_schemes(args)}
    for record in records:
        query, target = record.pop("chains")
        record["schemes"] = evaluate_pair(query, target, record["reference"], schemes)
        record.pop("reference")
    if not records:
        sys.exit("no usable pairs")
    with_reference = [r for r in records if r["reference_tm"] is not None]
    print(f"\n{len(records)} pairs; {len(with_reference)} with a US-align reference"
          + (f", median reference TM={median([r['reference_tm'] for r in with_reference]):.2f}"
             if with_reference else ""))

    columns = list(schemes)
    bins = [(label, [r for r in records if predicate(r["identity"])])
            for label, predicate in IDENTITY_BINS]
    print_table("MEDIAN TM-score of the superposition each alignment implies", bins, columns,
                lambda sel, c: median([r["schemes"][c]["tm"] for r in sel]))
    if with_reference:
        ref_bins = [(label, [r for r in with_reference if predicate(r["identity"])])
                    for label, predicate in IDENTITY_BINS]
        print_table("MEDIAN residue-pair F1 against US-align", ref_bins, columns,
                    lambda sel, c: median([r["schemes"][c]["f1"] for r in sel]))
        print_table("MEDIAN residue-pair F1 (+/- 4 residue shift allowed)", ref_bins, columns,
                    lambda sel, c: median([r["schemes"][c]["f1_shift4"] for r in sel]))
    print_table("MEDIAN alignment coverage of the shorter chain", bins, columns,
                lambda sel, c: median([r["schemes"][c]["coverage"] for r in sel]))
    print_table("MEDIAN identity over the aligned region", bins, columns,
                lambda sel, c: median([r["schemes"][c]["identity"] for r in sel]),
                note="structurally correct alignments span more and so report lower identity")

    if args.output:
        json.dump(records, open(args.output, "w"), indent=1)
        print(f"\nwrote {args.output}")


def command_sweep(args):
    records = collect(args, need_reference=True)
    if not records:
        sys.exit("no usable pairs with a US-align reference")
    schemes = [s for s in args.schemes]
    rng = random.Random(args.seed + 1)
    rows, baseline = [], []
    for n, record in enumerate(records):
        query, target = record["chains"]
        reference, length_norm = record["reference"], min(len(query), len(target))
        pairs, _ = align(query, target, SCHEMES["blosum_sw"])
        baseline.append(dict(identity=record["identity"], f1=prf(pairs, reference["pairs"])[2],
                             tm=tm_score(pairs, query, target, length_norm) if n < args.tm_limit else None))
        cells = {}
        for mode in args.modes:
            for level in args.levels:
                degraded = degrade_3di(query, level, mode, rng)
                for scheme in schemes:
                    predicted, _ = align(degraded, target, SCHEMES[scheme])
                    cells[(mode, level, scheme)] = dict(
                        f1=prf(predicted, reference["pairs"])[2],
                        tm=tm_score(predicted, query, target, length_norm) if n < args.tm_limit else None,
                    )
        rows.append(dict(identity=record["identity"], cells=cells))
        if len(rows) % 50 == 0:
            print(f"  {len(rows)} pairs swept", flush=True)

    low = [r for r in rows if r["identity"] < 0.20]
    print(f"\n{len(rows)} pairs ({len(low)} below 20% identity)")
    print(f"baseline BLOSUM62 local: median F1={median([b['f1'] for b in baseline]):.3f}"
          f" (<20% identity: {median([b['f1'] for b in baseline if b['identity'] < 0.20]):.3f}),"
          f" median TM={median([b['tm'] for b in baseline]):.3f}")
    for metric, title in (("f1", "residue-pair F1 against US-align"),
                          ("tm", "TM-score of the implied superposition")):
        for subset, label in ((rows, "all pairs"), (low, "pairs below 20% identity")):
            if not subset:
                continue
            print(f"\nMEDIAN {title}, by simulated query-3Di accuracy ({label})")
            print(f"{'scheme':<12}{'errors':<12}" + "".join(f"{int(l * 100):>8}%" for l in args.levels))
            for scheme in schemes:
                for mode in args.modes:
                    cells = "".join(
                        f"{median([r['cells'][(mode, level, scheme)][metric] for r in subset]):>9.3f}"
                        for level in args.levels)
                    print(f"{scheme:<12}{mode:<12}{cells}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(sub):
        sub.add_argument("--usalign", default=None,
                         help="path to the US-align binary (enables reference-based F1)")

    pair = subparsers.add_parser("pair", help="compare two chains")
    pair.add_argument("--query", required=True, help="query structure (PDB or mmCIF, optionally .gz)")
    pair.add_argument("--target", required=True, help="target structure")
    pair.add_argument("--query-chain", default=None)
    pair.add_argument("--target-chain", default=None)
    add_common(pair)
    pair.set_defaults(func=command_pair)

    for name, help_text, func in (("benchmark", "run over a table of pairs", command_benchmark),
                                  ("sweep", "degrade the query 3Di and re-measure", command_sweep)):
        sub = subparsers.add_parser(name, help=help_text)
        sub.add_argument("--pairs", required=True,
                         help="CSV of id1,id2,tm[,tm2] (e.g. cath_23M.csv, TMfast.dual.csv)")
        sub.add_argument("--structures", required=True, help="directory of structure files")
        sub.add_argument("--suffix", default=".pdb", help="structure file suffix [.pdb]")
        sub.add_argument("--limit", type=int, default=200, help="pairs to evaluate [200]")
        sub.add_argument("--oversample", type=int, default=4,
                         help="candidates sampled per evaluated pair [4]")
        sub.add_argument("--min-tm", type=float, default=0.7, help="minimum reference TM [0.7]")
        sub.add_argument("--min-length", type=int, default=60)
        sub.add_argument("--max-length", type=int, default=600)
        sub.add_argument("--line-rate", type=float, default=1.0,
                         help="fraction of table lines read, for very large tables [1.0]")
        sub.add_argument("--seed", type=int, default=20260916)
        sub.add_argument("--query-3di", default=None,
                         help="FASTA of predicted query 3Di (predict_three_di.py); pairs whose "
                              "query is absent from it are skipped")
        sub.add_argument("--query-profile", default=None,
                         help=".npz of predicted query 3Di profiles (predict_three_di.py "
                              "profiles.npz: (L, 20) log-probabilities keyed by domain); "
                              "the argmax replaces the query 3Di as --query-3di would")
        sub.add_argument("--profile-schemes", nargs="+", default=["3di_aa_sw"],
                         choices=[n for n, s in SCHEMES.items() if s["weights"] is not None],
                         help="schemes to rerun with the profile [3di_aa_sw]")
        sub.add_argument("--profile-mode", nargs="+", default=["argmax", "expected", "logodds"],
                         choices=list(PROFILE_TAGS),
                         help="3Di term: argmax row, expected score, or mixture log-odds "
                              "[argmax expected logodds]")
        sub.add_argument("--min-confidence", type=float, nargs="*", default=[],
                         help="also run each mode with the 3Di term dropped where max p < t")
        sub.add_argument("--profile-scale", type=int, default=100,
                         help="integer scale of profile scores and gaps; 1 reproduces the "
                              "string schemes' rounding exactly [100]")
        sub.add_argument("--target-profile", default=None,
                         help="predicted 3Di for the TARGETS too (sequence-only database): the "
                              "same profiles.npz format; pairs whose target is absent are skipped")
        sub.add_argument("--target-kernel", default=None,
                         help="target kernel from fit_target_kernel.py; adds a '+tk' variant of "
                              "each profile scheme that reads the target's confidence code")
        sub.add_argument("--profile-temperature", type=float, default=1.0,
                         help="softmax temperature applied to the stored log-probabilities [1.0]")
        add_common(sub)
        sub.set_defaults(func=func)

    benchmark = subparsers.choices["benchmark"]
    benchmark.add_argument("--output", default=None, help="write per-pair results as JSON")

    sweep = subparsers.choices["sweep"]
    sweep.add_argument("--levels", type=float, nargs="+", default=[1.0, 0.9, 0.8, 0.65, 0.5],
                       help="simulated query-3Di accuracies [1.0 0.9 0.8 0.65 0.5]")
    sweep.add_argument("--modes", nargs="+", default=["confusion", "uniform"],
                       choices=["confusion", "uniform"],
                       help="how wrong states are drawn [confusion uniform]")
    sweep.add_argument("--schemes", nargs="+", default=["3di_aa_sw", "3di_sw"],
                       choices=list(SCHEMES))
    sweep.add_argument("--tm-limit", type=int, default=100,
                       help="pairs for which the slower TM-score is computed [100]")

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
