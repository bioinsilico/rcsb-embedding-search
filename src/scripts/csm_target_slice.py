"""Computed structure models as search targets, with pLDDT masking.

Every benchmark so far used experimental domains on both sides.  A database like
RCSB.org's also holds computed structure models, whose 3Di is exact in the sense that it
comes from coordinates, but whose coordinates are a prediction - and in low-confidence
regions the local geometry, and so the 3Di, is not meaningful.  The brief's rule for
training labels (drop pLDDT < 70) applies to targets too, and this measures whether it
should.

Four versions of the same target, scored by the same query:

* ``domain``        the experimental domain, as in every earlier benchmark;
* ``chain``         the experimental full chain - CSMs are whole chains, so this is the
                    like-for-like comparison, and the first full-chain test here;
* ``csm``           the AlphaFold model of the same protein, 3Di from its coordinates;
* ``csm_masked``    the same, with pLDDT < threshold residues set to an "unknown" code
                    that scores on amino acids only (kernel row = the 3Di background).

It also reports how much of a model falls below the threshold, and how often the model's
3Di agrees with the experimental structure's, stratified by pLDDT - the statistic that
says whether a CSM's 3Di can be stored as if it were experimental.

    python csm_target_slice.py --pairs cath_test_pairs_capped.csv --domain-uniprot map.tsv \\
        --csm-dir csm --structures /data/cath_23M/pdb --entries /data/pdb \\
        --query-profile profiles.npz --usalign ./USalign
"""
from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
from dataclasses import replace

import numpy as np
from biotite.sequence import ProteinSequence
from biotite.sequence.align import SubstitutionMatrix, align_optimal
from biotite.structure.io.pdb import PDBFile

from structure_alignment import (
    SCHEMES, THREE_DI, UnreadableStructure, align, load_chain, prf, reference_alignment, tm_score,
    write_ca_pdb,
)
from structure_alignment_benchmark import read_profiles

BACKGROUND = np.array([0.0489372, 0.0306991, 0.101049, 0.0329671, 0.0276149, 0.0416262, 0.0452521,
                       0.030876, 0.0297251, 0.0607036, 0.0150238, 0.0215826, 0.0783843, 0.0512926,
                       0.0264886, 0.0610702, 0.0201311, 0.215998, 0.0310265, 0.0295417])
#: 20 one-hot rows plus an "unknown" row: a residue whose 3Di cannot be trusted scores 0
#: against every state (the background row is neutral in log-odds), i.e. amino acids only.
UNKNOWN_CODE = 20
KERNEL = np.vstack([np.eye(20), BACKGROUND / BACKGROUND.sum()])


def plddt_per_residue(path: str) -> np.ndarray:
    """pLDDT of each CA atom, in file order - AlphaFold writes it in the B-factor column."""
    atoms = PDBFile.read(path).get_structure(model=1, extra_fields=["b_factor"])
    return atoms.b_factor[atoms.atom_name == "CA"]


def ca_file(chain, directory: str) -> str:
    """Write a chain as a CA-only PDB for US-align.

    The reference has to be indexed the same way we score: handing US-align an mmCIF entry
    makes it read every chain, so its residue indices no longer match the single chain the
    scoring uses, and the resulting F1 is meaningless. Writing exactly the residues of this
    Chain removes the ambiguity.
    """
    path = os.path.join(directory, f"{chain.name.replace('/', '_')}.pdb")
    if not os.path.exists(path):
        write_ca_pdb(chain, path)
    return path


def coded(chain, codes):
    return type(chain)(chain.name, chain.sequence, chain.three_di, chain.ca_coord, None, codes)


def state_codes(chain) -> np.ndarray:
    return np.array([THREE_DI.index(c) if c in THREE_DI else THREE_DI.index("d") for c in chain.three_di])


def three_di_agreement(model, experimental) -> tuple[int, int, np.ndarray, np.ndarray]:
    """Align the two sequences and compare 3Di where they match; returns (same, total, plddt, agree)."""
    matrix = SubstitutionMatrix.std_protein_matrix()
    alignment = align_optimal(ProteinSequence(model.sequence), ProteinSequence(experimental.sequence),
                              matrix, gap_penalty=(-10, -1), local=False, terminal_penalty=False,
                              max_number=1)[0]
    trace = alignment.trace
    paired = trace[(trace[:, 0] != -1) & (trace[:, 1] != -1)]
    same = sum(model.three_di[i] == experimental.three_di[j] for i, j in paired)
    return same, len(paired), paired[:, 0], np.array(
        [model.three_di[i] == experimental.three_di[j] for i, j in paired])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--domain-uniprot", required=True, help="TSV: target domain -> UniProt accession")
    parser.add_argument("--csm-dir", required=True)
    parser.add_argument("--structures", required=True, help="directory of domain .pdb files")
    parser.add_argument("--entries", required=True, help="directory of entry .cif.gz files")
    parser.add_argument("--query-profile", required=True)
    parser.add_argument("--profile-temperature", type=float, default=1.0)
    parser.add_argument("--profile-scale", type=int, default=100)
    parser.add_argument("--mode", default="logodds", choices=["argmax", "expected", "logodds"])
    parser.add_argument("--usalign", required=True)
    parser.add_argument("--weight-amino", type=float, default=1.4)
    parser.add_argument("--weight-3di", type=float, default=2.1)
    parser.add_argument("--gap-open", type=float, default=30)
    parser.add_argument("--gap-extend", type=float, default=3)
    parser.add_argument("--plddt", type=float, default=70.0)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--max-length", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    accession = dict(line.split() for line in open(args.domain_uniprot))
    profiles = read_profiles(args.query_profile, args.profile_temperature)
    scheme = dict(SCHEMES["3di_aa_sw"], weights=(args.weight_amino, args.weight_3di),
                  gaps=(-args.gap_open, -args.gap_extend), profile=args.mode,
                  scale=args.profile_scale, target_kernel=KERNEL)

    rows = [line.strip().split(",") for line in open(args.pairs)]
    random.Random(args.seed).shuffle(rows)
    records, statistics = [], []
    workspace = tempfile.mkdtemp(prefix="csm_slice_")
    for query_id, target_id, _ in rows:
        if len(records) >= args.limit:
            break
        if query_id not in profiles or target_id not in accession:
            continue
        csm_path = os.path.join(args.csm_dir, f"AF-{accession[target_id]}.pdb")
        query_path = os.path.join(args.structures, query_id + ".pdb")
        domain_path = os.path.join(args.structures, target_id + ".pdb")
        entry_path = os.path.join(args.entries, f"{target_id[:4].upper()}.cif.gz")
        if not all(os.path.exists(p) for p in (csm_path, query_path, domain_path, entry_path)):
            continue
        try:
            query = load_chain(query_path, name=query_id)
            domain = load_chain(domain_path, name=target_id)
            chain = load_chain(entry_path, chain_id=target_id[4], name=target_id + "_chain")
            csm = load_chain(csm_path, name=accession[target_id])
        except (UnreadableStructure, ValueError, KeyError, IndexError):
            continue
        if len(profiles[query_id]) != len(query) or max(len(chain), len(csm)) > args.max_length:
            continue
        plddt = plddt_per_residue(csm_path)
        if len(plddt) != len(csm):
            continue

        same, total, positions, agree = three_di_agreement(csm, chain)
        statistics.append(dict(target=target_id, accession=accession[target_id],
                               csm_length=len(csm), chain_length=len(chain),
                               low_plddt=float((plddt < args.plddt).mean()),
                               mean_plddt=float(plddt.mean()),
                               agreement=same / max(total, 1), aligned=total,
                               agree_high=float(agree[plddt[positions] >= args.plddt].mean())
                               if (plddt[positions] >= args.plddt).any() else float("nan"),
                               agree_low=float(agree[plddt[positions] < args.plddt].mean())
                               if (plddt[positions] < args.plddt).any() else float("nan")))

        query_ca = ca_file(query, workspace)
        query = replace(query, three_di_profile=profiles[query_id])
        codes = state_codes(csm)
        versions = {
            "domain": coded(domain, state_codes(domain)),
            "chain": coded(chain, state_codes(chain)),
            "csm": coded(csm, codes),
            "csm_masked": coded(csm, np.where(plddt < args.plddt, UNKNOWN_CODE, codes)),
        }
        record = dict(query=query_id, target=target_id, versions={})
        for name, target in versions.items():
            reference = reference_alignment(args.usalign, query_ca, ca_file(target, workspace))
            if reference is None or not len(reference.get("pairs", [])):
                continue
            pairs, _ = align(query, target, scheme)
            precision, recall, f1 = prf(pairs, reference["pairs"])
            record["versions"][name] = dict(
                f1=f1, precision=precision, recall=recall, n_aligned=int(len(pairs)),
                tm=tm_score(pairs, query, target, min(len(query), len(target))),
                reference_pairs=int(len(reference["pairs"])), target_length=len(target))
        if len(record["versions"]) == 4:
            records.append(record)
            if len(records) % 25 == 0:
                print(f"  {len(records)} pairs", flush=True)

    print(f"\n{len(records)} pairs with all four target versions; {len(statistics)} CSM/experimental comparisons")
    low = np.array([s["low_plddt"] for s in statistics])
    print(f"CSM pLDDT: mean {np.mean([s['mean_plddt'] for s in statistics]):.1f}; "
          f"residues below {args.plddt:.0f}: {low.mean():.1%} on average, "
          f"{(low > 0.2).mean():.0%} of models have more than 20%")
    print(f"3Di agreement, CSM vs experimental chain of the same protein: "
          f"{np.nanmean([s['agreement'] for s in statistics]):.1%} overall, "
          f"{np.nanmean([s['agree_high'] for s in statistics]):.1%} at pLDDT >= {args.plddt:.0f}, "
          f"{np.nanmean([s['agree_low'] for s in statistics]):.1%} below")
    print(f"CSM length vs experimental chain: {np.mean([s['csm_length'] for s in statistics]):.0f} vs "
          f"{np.mean([s['chain_length'] for s in statistics]):.0f} residues")

    print(f"\n{'target version':<14}{'mean F1':>9}{'precision':>11}{'recall':>9}{'TM':>8}"
          f"{'aligned':>9}{'target len':>12}")
    for name in ("domain", "chain", "csm", "csm_masked"):
        values = {k: np.array([r["versions"][name][k] for r in records])
                  for k in ("f1", "precision", "recall", "tm", "n_aligned", "target_length")}
        print(f"{name:<14}{values['f1'].mean():>9.4f}{values['precision'].mean():>11.4f}"
              f"{values['recall'].mean():>9.4f}{values['tm'].mean():>8.4f}"
              f"{values['n_aligned'].mean():>9.1f}{values['target_length'].mean():>12.0f}")

    base = np.array([r["versions"]["csm"]["f1"] for r in records])
    masked = np.array([r["versions"]["csm_masked"]["f1"] for r in records])
    rng = np.random.default_rng(0)
    delta = masked - base
    boot = delta[rng.integers(0, len(delta), (4000, len(delta)))].mean(1)
    print(f"\npLDDT masking on CSM targets: {delta.mean():+.4f} mean F1 "
          f"[{np.percentile(boot, 2.5):+.4f}, {np.percentile(boot, 97.5):+.4f}]")

    if args.output:
        with open(args.output, "w") as handle:
            json.dump(dict(plddt=args.plddt, records=records, statistics=statistics), handle)


if __name__ == "__main__":
    main()
