"""Split CATH + SCOP domains into train / val / test by holding out whole families.

Families are CATH homologous superfamilies (C.A.T.H, 4th level) and SCOP
superfamilies (3rd level).  Holding out whole families makes validation answer
the question the 3Di head actually faces: does it transfer to folds it has not
been trained on?

The two databases classify many of the same proteins, so a CATH family cannot
be held out on its own: the same residues would reach training as a SCOP domain
(and vice versa).  Families are therefore linked across databases, and linked
families are held out together as one *component*.  Two families are linked when
a CATH domain and a SCOP domain from the same PDB chain share more than
``--min-overlap`` of the shorter domain's residues.

Residue overlap matters here, not just a shared chain.  On multi-domain chains a
chain-level link joins unrelated families: linking by chain puts ~57% of all
domains into one component, while a 50% residue-overlap link leaves the largest
component at ~9%.

Components are shuffled and assigned to test, then val, until each reaches its
share of domains.  Components larger than ``--max-component-share`` of a split's
budget are kept in training, so one huge family cannot fill a split by itself.

    python domain_family_split.py \\
        --cath-pdb .../cath-pdb --cath-classes .../cath_domain_ids.tsv \\
        --scop-pdb .../scop-pdb --scop-classes .../scop-zenodo_class.tsv \\
        --output domain_split.tsv
"""
from __future__ import annotations

import argparse
import os
import random
from collections import Counter, defaultdict
from multiprocessing import Pool

SPLITS = ("train", "val", "test")


def read_residue_ids(path: str) -> frozenset[str]:
    """Residue number + insertion code of every CA atom in the first model.

    A plain text scan rather than a structure parser: it only has to decide
    which residues two domain files share, over ~47k files.  Chain letters are
    ignored because the SCOP files leave the chain column blank.
    """
    residues = set()
    try:
        with open(path) as handle:
            for line in handle:
                if line.startswith("ENDMDL"):
                    break
                if line.startswith(("ATOM", "HETATM")) and line[12:16] == " CA ":
                    residues.add(line[22:27])
    except OSError:
        pass
    return frozenset(residues)


def _read_job(job: tuple[str, str]) -> tuple[str, frozenset[str]]:
    domain, path = job
    return domain, read_residue_ids(path)


def load_classes(path: str, levels: int) -> dict[str, str]:
    """``domain -> family`` from a whitespace-separated ``id classification`` file."""
    families = {}
    with open(path) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) >= 2:
                families[fields[0]] = ".".join(fields[1].split(".")[:levels])
    return families


def list_domains(directory: str) -> dict[str, str]:
    return {
        name[:-4]: os.path.join(directory, name)
        for name in os.listdir(directory) if name.endswith(".pdb")
    }


def chain_key(domain: str, source: str) -> str:
    """PDB id + chain letter, case-folded: CATH ``1oaiA00``, SCOP ``d1dr9a1``.

    SCOP writes the chain in lower case and ``.`` for domains spanning several
    chains; those match every chain of the entry (see ``candidate_keys``).
    """
    if source == "cath":
        return (domain[:4] + domain[4]).lower()
    return (domain[1:5] + domain[5]).lower()


class UnionFind:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a: str, b: str) -> None:
        self.parent[self.find(a)] = self.find(b)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cath-pdb", required=True, help="directory of CATH domain .pdb files")
    parser.add_argument("--cath-classes", required=True, help="cath_domain_ids.tsv: domain C.A.T.H")
    parser.add_argument("--scop-pdb", required=True, help="directory of SCOP domain .pdb files")
    parser.add_argument("--scop-classes", required=True, help="scop class file: domain a.b.c.d")
    parser.add_argument("--output", required=True, help="TSV manifest to write")
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--test-fraction", type=float, default=0.05)
    parser.add_argument("--min-overlap", type=float, default=0.5,
                        help="residue share of the shorter domain that links two families")
    parser.add_argument("--max-component-share", type=float, default=0.2,
                        help="components larger than this share of a split's budget stay in train")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ---- domains and their families -------------------------------------------------
    sources = {
        "cath": (list_domains(args.cath_pdb), load_classes(args.cath_classes, levels=4)),
        "scop": (list_domains(args.scop_pdb), load_classes(args.scop_classes, levels=3)),
    }
    domains: dict[str, dict] = {}
    for source, (paths, families) in sources.items():
        missing = sorted(d for d in paths if d not in families)
        if missing:
            print(f"{source}: {len(missing):,} files without a class are skipped, e.g. {missing[:3]}")
        for domain, path in paths.items():
            if domain in families:
                if domain in domains:
                    raise SystemExit(f"domain id {domain} appears in both databases")
                domains[domain] = dict(source=source, path=path, family=f"{source}:{families[domain]}")

    with Pool(args.workers) as pool:
        residues = dict(pool.map(_read_job, [(d, v["path"]) for d, v in domains.items()], chunksize=64))

    # ---- link families through residue-overlapping CATH / SCOP domains -------------
    families = UnionFind()
    for info in domains.values():
        families.find(info["family"])

    cath_by_chain = defaultdict(list)
    cath_by_entry = defaultdict(list)
    for domain, info in domains.items():
        if info["source"] == "cath":
            cath_by_chain[chain_key(domain, "cath")].append(domain)
            cath_by_entry[domain[:4].lower()].append(domain)

    n_links = 0
    near_misses = []   # overlapping but below threshold: not linked, checked after the split
    for domain, info in domains.items():
        if info["source"] != "scop" or not residues[domain]:
            continue
        key = chain_key(domain, "scop")
        candidates = cath_by_entry[key[:4]] if key[4] == "." else cath_by_chain[key]
        for cath_domain in candidates:
            if not residues[cath_domain]:
                continue
            shared = len(residues[domain] & residues[cath_domain])
            overlap = shared / min(len(residues[domain]), len(residues[cath_domain]))
            if overlap > args.min_overlap:
                families.union(info["family"], domains[cath_domain]["family"])
                n_links += 1
            elif shared:
                near_misses.append((domain, cath_domain, overlap))

    by_root = defaultdict(list)
    for domain, info in domains.items():
        by_root[families.find(info["family"])].append(domain)
    # Name components by size rank (c00000 is the largest); the union-find root is arbitrary.
    component_members = {}
    for rank, members in enumerate(sorted(by_root.values(), key=lambda m: (-len(m), min(m)))):
        component_members[f"c{rank:05d}"] = members
        for domain in members:
            domains[domain]["component"] = f"c{rank:05d}"

    # ---- assign components to splits -------------------------------------------------
    total = len(domains)
    budgets = {"test": int(args.test_fraction * total), "val": int(args.val_fraction * total)}
    order = sorted(component_members)
    random.Random(args.seed).shuffle(order)

    assignment = {}
    filled = Counter()
    too_large = 0
    for component in order:
        size = len(component_members[component])
        for split in ("test", "val"):
            if size > args.max_component_share * budgets[split]:
                continue
            if filled[split] + size <= budgets[split]:
                assignment[component] = split
                filled[split] += size
                break
        else:
            too_large += size > args.max_component_share * min(budgets.values())
            assignment[component] = "train"

    for info in domains.values():
        info["split"] = assignment[info["component"]]

    with open(args.output, "w") as handle:
        handle.write("domain_id\tsource\tfamily\tcomponent\tsplit\tn_residues\tpath\n")
        for domain in sorted(domains):
            info = domains[domain]
            handle.write(f"{domain}\t{info['source']}\t{info['family']}\t{info['component']}\t"
                         f"{info['split']}\t{len(residues[domain])}\t{info['path']}\n")

    # ---- report ------------------------------------------------------------------------
    sizes = sorted((len(m) for m in component_members.values()), reverse=True)
    n_families = len({info["family"] for info in domains.values()})
    print(f"domains    : {total:,}  ({sum(1 for d in domains.values() if d['source'] == 'cath'):,} CATH, "
          f"{sum(1 for d in domains.values() if d['source'] == 'scop'):,} SCOP; "
          f"{sum(1 for d in domains if not residues[d]):,} without CA atoms)")
    print(f"families   : {n_families:,} linked by {n_links:,} overlapping domain pairs into "
          f"{len(sizes):,} components")
    print(f"components : largest {sizes[:5]} ({100 * sizes[0] / total:.1f}% of domains); "
          f"{too_large:,} too large to hold out")
    print(f"\n{'split':<6} {'domains':>8} {'share':>7} {'CATH':>7} {'SCOP':>7} {'families':>9} {'components':>11}")
    for split in SPLITS:
        members = [d for d in domains.values() if d["split"] == split]
        print(f"{split:<6} {len(members):>8,} {len(members) / total:>7.1%} "
              f"{sum(1 for d in members if d['source'] == 'cath'):>7,} "
              f"{sum(1 for d in members if d['source'] == 'scop'):>7,} "
              f"{len({d['family'] for d in members}):>9,} {len({d['component'] for d in members}):>11,}")

    crossing = [(s, c, o) for s, c, o in near_misses if domains[s]["split"] != domains[c]["split"]]
    print(f"\nresidue-overlapping CATH/SCOP pairs below the link threshold: {len(near_misses):,}; "
          f"{len(crossing):,} of them fall in different splits")
    for scop_domain, cath_domain, overlap in sorted(crossing, key=lambda x: -x[2])[:5]:
        print(f"  {scop_domain} ({domains[scop_domain]['split']}) ~ {cath_domain} "
              f"({domains[cath_domain]['split']}) overlap {overlap:.0%}")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
