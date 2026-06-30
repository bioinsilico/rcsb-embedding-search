"""Detect chains that biotite splits into disjoint segments.

biotite's ``get_chain_starts`` begins a new chain whenever the residue id
*decreases*, so a single chain id can resolve to several disjoint segments. In
the CATH data those extra segments are distinct entities sharing the chain id
(bound peptides flagged by an insertion code, His-tags, symmetry mates, or two
mislabelled chains) rather than a renumbered continuation of one chain.
Concatenating them feeds a chimeric, non-physical sequence to ESM, so callers
skip a chain id that splits into more than one segment.
"""
from __future__ import annotations

import biotite.structure.io.pdb as pdb
from biotite.structure import filter_amino_acids, get_chain_starts, get_residue_starts


def segment_chains_by_id(atom_array):
    """Group an atom array into ``{chain_id: [segment, ...]}``.

    Each segment is the slice between consecutive ``get_chain_starts`` indices,
    so a chain id with more than one segment is disjoint (a residue-id decrease
    split it). The caller applies whatever residue filtering it wants first; this
    only does the splitting.
    """
    segments = {}
    starts = get_chain_starts(atom_array, add_exclusive_stop=True)
    for i in range(len(starts) - 1):
        seg = atom_array[starts[i]:starts[i + 1]]
        segments.setdefault(str(seg.chain_id[0]), []).append(seg)
    return segments


def residue_count(seg):
    """Number of residues in a (segment) atom array."""
    return len(get_residue_starts(seg))


def disjoint_segment_sizes(pdb_file, chain_id=None):
    """Residue-count list if ``chain_id`` splits into >1 segment, else ``None``.

    Mirrors how ``esm``'s ``ProteinChain.from_pdb`` reads a structure: model 1,
    amino-acid + non-hetero atoms of a single (author) chain. ``chain_id=None``
    uses the first detected chain, matching ``ProteinChain``'s ``"detect"``.
    Returned sizes sum to the sequence length ``ProteinChain`` would build, so a
    non-``None`` result means that sequence would be a chimera of fused segments.
    """
    atom_array = pdb.PDBFile.read(pdb_file).get_structure(model=1)
    if atom_array.array_length() == 0:
        return None
    if chain_id is None:
        chain_id = str(atom_array.chain_id[0])
    atom_array = atom_array[
        filter_amino_acids(atom_array)
        & ~atom_array.hetero
        & (atom_array.chain_id == chain_id)
    ]
    if atom_array.array_length() == 0:
        return None
    segs = segment_chains_by_id(atom_array).get(chain_id, [])
    if len(segs) <= 1:
        return None
    return [residue_count(s) for s in segs]
