"""CATH superfamily labels, used to propose genuinely homologous pairs.

The minimizer index reaches related windows, but what it mostly surfaces are
near-duplicates: on CATH its high-scoring proposals sit at ~0.96 internal
identity.  The 15-60% identity band — remote homology, the case a search model
actually has to generalise over — stays thin, because two domains that diverged
that far rarely share an exact seed.

CATH already knows which domains are homologous.  ``cath-domain-list.txt``
assigns every domain a C.A.T.H classification, and two domains sharing all four
levels are, by CATH's own curation, the same homologous superfamily.  Proposing
pairs from within one superfamily therefore yields real remote homologs with no
simulation and no assumption about how sequences diverge.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def parse_cath_domain_id(header: str) -> str:
    """``cath|4_4_0|101mA00/0-153`` -> ``101mA00``."""
    parts = header.split('|')
    tail = parts[2] if len(parts) >= 3 else parts[-1]
    return tail.split('/')[0]


def load_superfamilies(path: str) -> dict[str, int]:
    """Map each domain id to a dense integer id for its C.A.T.H superfamily.

    The file is whitespace-separated with ``#`` comments; columns 2-5 are the
    class, architecture, topology and homologous-superfamily numbers.
    """
    families: dict[tuple, int] = {}
    domains: dict[str, int] = {}
    with open(path) as handle:
        for line in handle:
            if line.startswith('#'):
                continue
            fields = line.split()
            if len(fields) < 5:
                continue
            key = (fields[1], fields[2], fields[3], fields[4])
            family = families.get(key)
            if family is None:
                family = len(families)
                families[key] = family
            domains[fields[0]] = family
    logger.info(
        f"Loaded {len(domains):,} CATH domains in {len(families):,} superfamilies "
        f"from {path}"
    )
    return domains


def load_cath_levels(path: str) -> dict[str, tuple]:
    """Map each domain to its full ``(C, A, T, H)`` classification.

    The four levels give progressively harder negatives.  Two domains sharing
    C.A.T but not H have the same fold and differ only in homology, so telling
    them apart cannot be done on amino-acid composition alone — which an
    untrained network already does well enough to reach 0.94 AUROC against
    random negatives.
    """
    levels: dict[str, tuple] = {}
    with open(path) as handle:
        for line in handle:
            if line.startswith('#'):
                continue
            f = line.split()
            if len(f) >= 5:
                levels[f[0]] = (f[1], f[2], f[3], f[4])
    return levels
