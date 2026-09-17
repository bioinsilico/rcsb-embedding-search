"""Amino acid sequence and masked Foldseek 3Di labels for one domain structure.

This is the single source of truth for what a training example of the
sequence -> 3Di model looks like.  The script that pre-computes ESM3 embeddings
and the Dataset that serves labels both call ``parse_domain``, so the sequence
ESM3 embedded is, by construction, the sequence the labels are indexed against.

Why not ``biotite.structure.alphabet.to_3di``: it returns ``.filled()`` states,
writing every residue the encoder could not describe as ``d``, which is also a
real state.  The encoder itself knows which residues those are (its output is a
masked array), so this module runs the same encoder steps and keeps the mask.

A residue is excluded from the loss for any of these reasons (bit flags, since
several can apply at once):

* ``TERMINUS``  - first or last residue of the domain.  The 3Di descriptors of
  residue i use CA(i-1) and CA(i+1), which do not exist there.
* ``BACKBONE``  - the encoder masked it: N, CA or C is missing on the residue,
  its sequence neighbours, or its structural partner and the partner's
  neighbours.  Foldseek propagates invalidity exactly this way.
* ``BREAK``     - a chain break (CA-CA distance above ``BREAK_DISTANCE``) sits
  next to the residue or next to its partner.  Foldseek does not detect breaks,
  so a descriptor spanning one is computed from two unconnected pieces.  CATH
  domains assembled from several chain segments hit this often.

Residues without a CA atom are kept in the sequence (ESM3 sees the full
observed sequence) and are always excluded, via ``BACKBONE``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import biotite.structure as struc
import biotite.structure.info as struc_info
from biotite.structure.alphabet.encoder import Encoder
from biotite.structure.io.pdb import PDBFile
from biotite.structure.util import coord_for_atom_name_per_residue

#: 3Di states in biotite's order; a label value k is the state ``THREE_DI[k]``.
THREE_DI = "acdefghiklmnpqrstvwy"
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

#: CA-CA distance (A) above which consecutive residues are treated as unconnected.
#: Trans peptides sit at ~3.8 A and cis at ~2.9 A.
BREAK_DISTANCE = 4.2

TERMINUS = 1
BACKBONE = 2
BREAK = 4
EXCLUSION_REASONS = {TERMINUS: "terminus", BACKBONE: "backbone", BREAK: "break"}

_ENCODER: Encoder | None = None


class UnusableDomain(Exception):
    """A structure file that cannot yield one clean chain."""


@dataclass
class DomainLabels:
    """One domain as index-aligned per-residue arrays."""

    name: str
    sequence: str            # one letter per residue, non-standard mapped to parent or X
    three_di: np.ndarray     # (L,) uint8 state index into THREE_DI, meaningless where not mask
    mask: np.ndarray         # (L,) bool, True where the label is usable for training
    exclusion: np.ndarray    # (L,) uint8 bit flags, 0 where mask is True
    ca_coord: np.ndarray     # (L, 3) float32, NaN where the residue has no CA

    def __len__(self) -> int:
        return len(self.sequence)

    @property
    def three_di_string(self) -> str:
        """3Di states with excluded residues written as '-'."""
        return "".join(THREE_DI[s] if m else "-" for s, m in zip(self.three_di, self.mask))


def _encoder() -> Encoder:
    # Loading the network weights costs more than encoding a domain, so do it once
    # per process (each DataLoader worker builds its own).
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = Encoder()
    return _ENCODER


def _one_letter(res_name: str) -> str:
    letter = struc_info.one_letter_code(res_name)
    return letter if letter is not None and letter in AMINO_ACIDS else "X"


def read_domain_atoms(path: str) -> struc.AtomArray:
    """Amino acid atoms of the single chain in a domain PDB file.

    Raises ``UnusableDomain`` for empty files, files with more than one chain id,
    and chain ids that biotite splits into disjoint segments (a residue-id
    decrease).  In the CATH/SCOP domain sets those segments are distinct entities
    sharing a chain id, so concatenating them would give ESM3 a chimeric sequence.
    """
    try:
        atoms = PDBFile.read(path).get_structure(model=1)
    except ValueError as error:          # "The file has 0 models"
        raise UnusableDomain(f"{path}: {error}") from None
    atoms = atoms[struc.filter_amino_acids(atoms)]
    if atoms.array_length() == 0:
        raise UnusableDomain(f"{path}: no amino acid residues")
    chain_ids = np.unique(atoms.chain_id)
    if len(chain_ids) > 1:
        raise UnusableDomain(f"{path}: {len(chain_ids)} chain ids {list(chain_ids)}")
    if len(struc.get_chain_starts(atoms)) > 1:
        raise UnusableDomain(f"{path}: chain splits into disjoint segments")
    return atoms


def sequence_from_atoms(atoms: struc.AtomArray) -> str:
    """One letter per residue, in file order: the sequence ESM3 embeds."""
    return "".join(_one_letter(r) for r in struc.get_residues(atoms)[1])


def domain_sequence(path: str) -> str:
    """The sequence ``parse_domain`` would return, without computing 3Di."""
    return sequence_from_atoms(read_domain_atoms(path))


def parse_domain(path: str, name: str | None = None) -> DomainLabels:
    """Read a domain PDB file into its sequence, 3Di labels and loss mask."""
    atoms = read_domain_atoms(path)
    sequence = sequence_from_atoms(atoms)

    ca, cb, n, c = coord_for_atom_name_per_residue(atoms, ["CA", "CB", "N", "C"])
    length = len(sequence)
    if ca.shape[0] != length:
        raise UnusableDomain(f"{path}: {ca.shape[0]} coordinate rows for {length} residues")
    if length < 3:
        raise UnusableDomain(f"{path}: only {length} residues")

    # Same steps as biotite's Encoder.encode, keeping the partner index and mask.
    encoder = _encoder()
    features = encoder.feature_encoder
    virtual_center = features.vc_encoder.encode(ca, cb, n, c)
    partner = features.partner_index_encoder._find_residue_partners(virtual_center)
    descriptors = features._calc_conformation_descriptors(ca, partner)
    encoder_mask = features._create_descriptor_mask(virtual_center.mask[:, 0], partner)[:, 0]
    states = encoder.vae_encoder(descriptors).astype(np.uint8)

    exclusion = np.zeros(length, dtype=np.uint8)
    exclusion[[0, -1]] |= TERMINUS
    exclusion[encoder_mask & (exclusion & TERMINUS == 0)] |= BACKBONE

    # A break between k and k+1 corrupts the descriptors of any residue i whose
    # own step vectors (i-1 -> i, i -> i+1) or whose partner's step vectors cross it.
    step = np.linalg.norm(np.diff(ca, axis=0), axis=1)          # NaN if a CA is missing
    broken_step = np.nan_to_num(step, nan=0.0) > BREAK_DISTANCE  # (L-1,) break after k
    touches_break = np.zeros(length, dtype=bool)
    touches_break[:-1] |= broken_step
    touches_break[1:] |= broken_step
    inner = np.arange(1, length - 1)
    corrupted = np.zeros(length, dtype=bool)
    corrupted[inner] = touches_break[inner] | touches_break[partner[inner]]
    exclusion[corrupted] |= BREAK

    return DomainLabels(
        name=name or path,
        sequence=sequence,
        three_di=states,
        mask=exclusion == 0,
        exclusion=exclusion,
        ca_coord=ca.astype(np.float32),
    )
