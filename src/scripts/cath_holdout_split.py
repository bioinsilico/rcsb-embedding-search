"""Pick whole CATH superfamilies to hold out of training.

Validating on fresh pairs of *domains the model trained on* measures very
little: the superfamily sampler draws families uniformly, so a small family's
domains are seen hundreds of times and the model can memorise them.  Holding out
entire superfamilies instead makes the validation score answer the question a
search model actually faces — does this transfer to a fold never seen before?

Families are chosen at random until a target share of domains is reached, and
oversized families are skipped so one giant superfamily cannot swallow the whole
budget (CATH's largest holds 49,516 domains against a median of 7).
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict

from dataset.utils.cath_superfamily import load_superfamilies, parse_cath_domain_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain-list', required=True, help='cath-domain-list.txt')
    parser.add_argument('--fasta', required=True, help='training FASTA')
    parser.add_argument('--output', required=True, help='file of held-out domain ids')
    parser.add_argument('--fraction', type=float, default=0.03,
                        help='target share of domains to hold out (default 0.03)')
    parser.add_argument('--min-family-size', type=int, default=2,
                        help='families smaller than this cannot form a pair (default 2)')
    parser.add_argument('--max-family-share', type=float, default=0.2,
                        help='skip families larger than this share of the budget')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    superfamily = load_superfamilies(args.domain_list)

    members = defaultdict(list)
    n_headers = n_mapped = 0
    with open(args.fasta) as handle:
        for line in handle:
            if not line.startswith('>'):
                continue
            n_headers += 1
            domain = parse_cath_domain_id(line[1:].strip().split()[0])
            family = superfamily.get(domain)
            if family is None:
                continue
            n_mapped += 1
            members[family].append(domain)

    budget = int(args.fraction * n_mapped)
    cap = max(1, int(args.max_family_share * budget))
    eligible = [f for f, m in members.items() if len(m) >= args.min_family_size]
    random.Random(args.seed).shuffle(eligible)

    chosen: list[int] = []
    held: list[str] = []
    skipped_large = 0
    for family in eligible:
        size = len(members[family])
        if size > cap:
            skipped_large += 1
            continue
        if len(held) + size > budget:
            continue
        chosen.append(family)
        held.extend(members[family])

    with open(args.output, 'w') as handle:
        handle.write('\n'.join(sorted(held)) + '\n')

    print(f"FASTA headers            : {n_headers:,} ({n_mapped:,} mapped to a superfamily)")
    print(f"families                 : {len(members):,} ({len(eligible):,} with >= "
          f"{args.min_family_size} domains)")
    print(f"budget                   : {budget:,} domains ({100 * args.fraction:.1f}%), "
          f"per-family cap {cap:,} ({skipped_large:,} families skipped as too large)")
    print(f"held out                 : {len(held):,} domains in {len(chosen):,} families "
          f"({100 * len(held) / n_mapped:.2f}% of the corpus)")
    print(f"remaining for training   : {n_mapped - len(held):,} domains in "
          f"{len(members) - len(chosen):,} families")
    print(f"written to               : {args.output}")


if __name__ == '__main__':
    main()
