"""Building blocks for judging sequence-level alignments against structure.

The question these tools answer: how close does a pairwise alignment computed from
sequence-level information come to the alignment implied by a structural superposition?

A structure contributes three parallel, index-aligned arrays (one entry per residue with a
CA atom): the amino acid sequence, the Foldseek 3Di structural-alphabet string, and the CA
coordinates. Alignments are then scored with a substitution matrix over the combined
(amino acid, 3Di) alphabet, so a single dynamic-programming call can weight sequence and
structure signal in any proportion.

Two measures of alignment quality:

* ``tm_score`` superimposes the two structures using only the residue pairs an alignment
  proposes and returns the TM-score of that superposition. It needs no reference alignment
  (the idea behind ``USalign -I``) and matches US-align to within 0.001 in our tests.
* ``reference_alignment`` runs US-align to obtain a structural reference, and ``prf``
  scores a candidate alignment against it as precision / recall / F1 over residue pairs.

``degrade_3di`` simulates an imperfect 3Di predictor, so the benefit of a predicted (rather
than exact) 3Di string can be estimated before training one.

A query can also carry a 3Di *profile*: per residue, a probability for each of the 20 states
(``Chain.three_di_profile``). Schemes with a ``profile`` key then score query residue i
against target symbol (a, s) with a position-specific row (``profile_matrix``), which
reproduces the string path exactly for a one-hot profile at ``scale=1``.

This module is imported by ``structure_alignment_benchmark.py``, which sits next to it.
"""
from __future__ import annotations

import gzip
import random
import re
import subprocess
from dataclasses import dataclass, replace

import numpy as np
import biotite.structure as struc
from biotite.sequence import Alphabet, GeneralSequence, ProteinSequence, PurePositionalSequence
from biotite.sequence.align import SubstitutionMatrix, align_optimal
from biotite.structure.alphabet import to_3di
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdbx import CIFFile, get_structure

# Amino acids in BLOSUM62 order plus X; 3Di states as biotite emits them (lower case).
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWYX"
THREE_DI = "acdefghiklmnpqrstvwy"
PAIR_ALPHABET = Alphabet(list(range(len(AMINO_ACIDS) * len(THREE_DI))))


def pair_alphabet(n_codes: int = len(THREE_DI)) -> Alphabet:
    """Alphabet of (amino acid, structural code) symbols; ``n_codes`` structural codes."""
    if n_codes == len(THREE_DI):
        return PAIR_ALPHABET
    key = ("pair_alphabet", n_codes)
    if key not in _MATRIX_CACHE:
        _MATRIX_CACHE[key] = Alphabet(list(range(len(AMINO_ACIDS) * n_codes)))
    return _MATRIX_CACHE[key]

# Foldseek's weighting of amino acid and 3Di substitution scores, and its gap penalties.
FOLDSEEK_WEIGHTS = (1.4, 2.1)
FOLDSEEK_GAPS = (-10, -1)
# From the header of biotite's matrix_data/3Di.mat: the 3Di matrix is in bit/2 units and
# lambda converts a score to natural-log odds.
LAMBDA_3DI = 0.351568

#: Scoring schemes compared by the benchmark. ``weights`` is (amino acid, 3Di); ``None``
#: means plain BLOSUM62 over the amino acid alphabet alone.
SCHEMES = {
    "blosum_sw": dict(weights=None, gaps=(-11, -1), local=True),
    "blosum_semi": dict(weights=None, gaps=(-11, -1), local=False),
    "3di_aa_sw": dict(weights=FOLDSEEK_WEIGHTS, gaps=FOLDSEEK_GAPS, local=True),
    "3di_aa_semi": dict(weights=FOLDSEEK_WEIGHTS, gaps=FOLDSEEK_GAPS, local=False),
    "3di_sw": dict(weights=(0.0, 2.1), gaps=FOLDSEEK_GAPS, local=True),
}

_MATRIX_CACHE: dict = {}


class UnreadableStructure(Exception):
    """A structure file that cannot be parsed into a single model."""


@dataclass
class Chain:
    """One chain as three index-aligned arrays, one entry per residue with a CA atom."""

    name: str
    sequence: str          # amino acids, one letter per residue
    three_di: str          # 3Di states, same length
    ca_coord: np.ndarray   # (L, 3) CA coordinates, same order
    three_di_profile: np.ndarray | None = None  # (L, 20) probabilities, THREE_DI order
    target_codes: np.ndarray | None = None      # (L,) codes into a target kernel, when this
                                                # chain is a target whose 3Di is predicted

    def __len__(self) -> int:
        return len(self.sequence)

    @property
    def invalid_3di_fraction(self) -> float:
        """Share of residues Foldseek could not encode.

        Foldseek writes these as 'd', which is also a valid state, so they cannot be told
        apart from a genuine 'd' afterwards. Residues at segment ends, with missing
        backbone atoms, or without a structural neighbour end up here.
        """
        return self.three_di.count("d") / max(1, len(self.three_di))

    def amino_codes(self) -> np.ndarray:
        return np.array([
            AMINO_ACIDS.index(c) if c in AMINO_ACIDS else AMINO_ACIDS.index("X")
            for c in self.sequence
        ])

    def pair_codes(self, n_codes: int = len(THREE_DI)) -> np.ndarray:
        """Encode residues as single symbols of the combined (amino acid, structural) alphabet.

        The structural part is the 3Di state by default. ``target_codes`` is used only when
        the caller asks for a wider alphabet (``n_codes`` beyond the 20 states), i.e. when a
        kernel interprets the codes; every other scheme reads the 3Di string as usual, which
        for a predicted target is its most likely state.
        """
        if self.target_codes is not None and n_codes != len(THREE_DI):
            structural = self.target_codes
        else:
            structural = np.array([
                THREE_DI.index(c) if c in THREE_DI else THREE_DI.index("d")
                for c in self.three_di
            ])
        return self.amino_codes() * n_codes + structural


def load_chain(path: str, chain_id: str | None = None, name: str | None = None) -> Chain:
    """Read one chain from a PDB or mmCIF file (optionally gzipped)."""
    if path.endswith((".cif", ".cif.gz")):
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt") as handle:
            atoms = get_structure(CIFFile.read(handle), model=1)
    else:
        pdb_file = PDBFile.read(path)
        try:
            atoms = pdb_file.get_structure(model=1)
        except ValueError as error:
            raise UnreadableStructure(f"{path}: {error}") from None
    atoms = atoms[struc.filter_amino_acids(atoms)]
    if chain_id is not None:
        atoms = atoms[atoms.chain_id == chain_id]
    if atoms.array_length() == 0:
        raise UnreadableStructure(f"{path}: no amino acid residues"
                                  f"{'' if chain_id is None else f' in chain {chain_id}'}")
    # One residue = one CA atom, so sequence, 3Di and coordinates stay index-aligned.
    ca_atoms = atoms[atoms.atom_name == "CA"]
    with_ca = set(zip(ca_atoms.chain_id, ca_atoms.res_id, ca_atoms.ins_code))
    atoms = atoms[np.array([
        (chain, res, ins) in with_ca
        for chain, res, ins in zip(atoms.chain_id, atoms.res_id, atoms.ins_code)
    ])]
    three_di_sequences, _ = to_3di(atoms)
    ca_atoms = atoms[atoms.atom_name == "CA"]
    sequence = "".join(ProteinSequence.convert_letter_3to1(r) for r in ca_atoms.res_name)
    three_di = "".join(str(s) for s in three_di_sequences)
    if not len(sequence) == len(three_di) == ca_atoms.array_length():
        raise UnreadableStructure(
            f"{path}: sequence ({len(sequence)}), 3Di ({len(three_di)}) and "
            f"CA ({ca_atoms.array_length()}) lengths disagree"
        )
    return Chain(name=name or path, sequence=sequence, three_di=three_di,
                 ca_coord=ca_atoms.coord.astype(np.float64))


def _base_scores() -> tuple[np.ndarray, np.ndarray]:
    """BLOSUM62 over AMINO_ACIDS (21 x 21) and the 3Di matrix over THREE_DI (20 x 20)."""
    if "base" not in _MATRIX_CACHE:
        blosum = SubstitutionMatrix.std_protein_matrix()
        blosum_alphabet = blosum.get_alphabet1()
        amino_scores = blosum.score_matrix()[np.ix_(
            [blosum_alphabet.encode(c) for c in AMINO_ACIDS],
            [blosum_alphabet.encode(c) for c in AMINO_ACIDS],
        )]
        structural = SubstitutionMatrix.std_3di_matrix()
        structural_alphabet = structural.get_alphabet1()
        structural_scores = structural.score_matrix()[np.ix_(
            [structural_alphabet.encode(c) for c in THREE_DI],
            [structural_alphabet.encode(c) for c in THREE_DI],
        )]
        _MATRIX_CACHE["base"] = (amino_scores, structural_scores)
    return _MATRIX_CACHE["base"]


def combined_matrix(weight_amino: float, weight_3di: float) -> SubstitutionMatrix:
    """Substitution matrix over (amino acid, 3Di) symbols, Foldseek-style weighted sum."""
    key = (weight_amino, weight_3di)
    if key in _MATRIX_CACHE:
        return _MATRIX_CACHE[key]
    amino_scores, structural_scores = _base_scores()
    scores = (weight_amino * amino_scores[:, None, :, None]
              + weight_3di * structural_scores[None, :, None, :])
    scores = np.rint(scores).astype(np.int32).reshape(len(PAIR_ALPHABET), len(PAIR_ALPHABET))
    _MATRIX_CACHE[key] = SubstitutionMatrix(PAIR_ALPHABET, PAIR_ALPHABET, scores)
    return _MATRIX_CACHE[key]


def profile_term(profile: np.ndarray, mode: str, kernel: np.ndarray | None = None) -> np.ndarray:
    """(L, C) 3Di score of each query position against each target code.

    ``kernel`` is (C, 20): the distribution over true states a target code stands for. The
    default is the identity, i.e. C = 20 exact states, which is a target whose 3Di is known.
    A predicted target instead stores a code per residue (its predicted state and confidence),
    and the kernel says what that code implies - so query uncertainty and target uncertainty
    are combined in one score:
        expected  sum_k sum_l p(k) q(l) M[k, l]
        logodds   (1 / lambda) ln sum_k sum_l p(k) q(l) exp(lambda M[k, l])

    ``argmax``   the most probable state's matrix row (the string path).
    ``expected`` sum_k p(k) * M[k, s], the expected substitution score.
    ``logodds``  (1 / lambda) * ln sum_k p(k) * exp(lambda * M[k, s]), the log-odds score of
                 the mixture; 0 for a profile equal to the matrix background.
    Both soft terms reduce to the argmax row, exactly, for a one-hot profile.
    """
    _, structural = _base_scores()
    p = profile / profile.sum(axis=1, keepdims=True)
    # scores[k, c]: state k against target code c. Identity kernel -> the matrix itself.
    scores = structural if kernel is None else structural @ kernel.T
    top = scores[p.argmax(axis=1)]
    if mode == "argmax":
        return top.astype(np.float64)
    if mode == "expected":
        return p @ scores
    if mode == "logodds":
        # Relative to the argmax row: log(1) = 0 keeps one-hot exact, and exponents stay
        # below lambda * 26 ~ 9. With a kernel the mixture is over (k, l) at once, which is
        # why the kernel is applied to exp(lambda M) rather than to M.
        if kernel is None:
            relative = np.exp(LAMBDA_3DI * (structural[None, :, :] - top[:, None, :]))
            return top + np.log(np.einsum("lk,lks->ls", p, relative)) / LAMBDA_3DI
        mixed = np.exp(LAMBDA_3DI * structural) @ kernel.T              # (20, C)
        return np.log(p @ mixed) / LAMBDA_3DI
    raise ValueError(f"unknown profile mode {mode!r}")


def profile_matrix(query: Chain, weights: tuple[float, float], mode: str, scale: int = 1,
                   min_confidence: float | None = None, kernel: np.ndarray | None = None):
    """Position-specific (L x 420) matrix for a query that carries a 3Di profile.

    Row i scores query residue i against every (amino acid, 3Di) target symbol. Where the
    profile's max probability is below ``min_confidence`` the 3Di term is dropped, leaving
    amino acids only. Scores are scaled by ``scale`` before rounding to integers.
    """
    weight_amino, weight_3di = weights
    amino_scores, _ = _base_scores()
    structural = profile_term(query.three_di_profile, mode, kernel)
    n_codes = structural.shape[1]
    if min_confidence is not None:
        low = query.three_di_profile.max(axis=1) < min_confidence
        structural = np.where(low[:, None], 0.0, structural)
    amino = query.amino_codes()
    # Same expression and operand order as combined_matrix, so a one-hot profile at scale 1
    # rounds bit-identically (x.5 ties included).
    scores = (weight_amino * amino_scores[amino][:, :, None]
              + weight_3di * structural[:, None, :]).reshape(len(query), len(AMINO_ACIDS) * n_codes)
    if scale != 1:
        scores = scale * scores
    query_seq = PurePositionalSequence(len(query))
    matrix = SubstitutionMatrix(query_seq.get_alphabet(), pair_alphabet(n_codes),
                                np.rint(scores).astype(np.int32))
    return query_seq, matrix


def align(query: Chain, target: Chain, scheme: dict) -> tuple[np.ndarray, float]:
    """Align two chains under one scheme; returns the aligned residue pairs and the score.

    A scheme with a ``profile`` key uses the query's 3Di profile (see ``profile_matrix``);
    its gaps are scaled with the scores and the score is reported unscaled.
    """
    scale = 1
    if scheme["weights"] is None:
        matrix = SubstitutionMatrix.std_protein_matrix()
        query_seq = ProteinSequence(query.sequence)
        target_seq = ProteinSequence(target.sequence)
    elif scheme.get("profile"):
        if query.three_di_profile is None:
            raise ValueError(f"scheme needs a query 3Di profile; {query.name} has none")
        scale = scheme.get("scale", 1)
        kernel = scheme.get("target_kernel")
        query_seq, matrix = profile_matrix(query, scheme["weights"], scheme["profile"], scale,
                                           scheme.get("min_confidence"), kernel)
        n_codes = len(THREE_DI) if kernel is None else kernel.shape[0]
        target_seq = GeneralSequence(pair_alphabet(n_codes), list(target.pair_codes(n_codes)))
    else:
        matrix = combined_matrix(*scheme["weights"])
        query_seq = GeneralSequence(PAIR_ALPHABET, list(query.pair_codes()))
        target_seq = GeneralSequence(PAIR_ALPHABET, list(target.pair_codes()))
    alignment = align_optimal(
        query_seq, target_seq, matrix,
        # Scaled penalties must stay Python ints: align_optimal type-checks them.
        gap_penalty=scheme["gaps"] if scale == 1 else tuple(int(round(g * scale)) for g in scheme["gaps"]),
        local=scheme["local"],
        terminal_penalty=False, max_number=1,
    )[0]
    trace = alignment.trace
    aligned = (trace[:, 0] != -1) & (trace[:, 1] != -1)
    return trace[aligned], alignment.score / scale if scale != 1 else int(alignment.score)


def tm_score(pairs: np.ndarray, query: Chain, target: Chain, length_norm: int) -> float:
    """TM-score of the superposition implied by a fixed alignment (as ``USalign -I``).

    Superimposes on seed fragments of the aligned pairs, then repeatedly re-superimposes on
    the pairs falling inside a distance cutoff, keeping the best TM-score over all seeds.
    """
    if len(pairs) < 3:
        return 0.0
    query_ca = query.ca_coord[pairs[:, 0]]
    target_ca = target.ca_coord[pairs[:, 1]]
    d0 = max(0.5, 1.24 * (length_norm - 15) ** (1 / 3) - 1.8)
    n_pairs = len(pairs)
    best = 0.0
    seeds = [n_pairs]
    while seeds[-1] > 4:
        seeds.append(max(4, seeds[-1] // 2))
    for seed in seeds:
        for start in range(0, n_pairs - seed + 1, max(1, seed // 2)):
            selected = np.arange(start, start + seed)
            for _ in range(20):
                if len(selected) < 3:
                    break
                _, transform = struc.superimpose(target_ca[selected], query_ca[selected])
                distances = np.linalg.norm(transform.apply(query_ca) - target_ca, axis=1)
                best = max(best, float(np.sum(1 / (1 + (distances / d0) ** 2)) / length_norm))
                cutoff = d0 + 1.0
                nearby = np.flatnonzero(distances < cutoff)
                while len(nearby) < 3 and cutoff < 20:
                    cutoff += 0.5
                    nearby = np.flatnonzero(distances < cutoff)
                if np.array_equal(nearby, selected):
                    break
                selected = nearby
    return best


def sequence_identity(pairs: np.ndarray, query: Chain, target: Chain) -> float:
    """Identical residues over aligned pairs."""
    if len(pairs) == 0:
        return 0.0
    return sum(query.sequence[i] == target.sequence[j] for i, j in pairs) / len(pairs)


def global_identity(query: Chain, target: Chain) -> float:
    """Identical residues over the shorter chain, from a semi-global BLOSUM62 alignment.

    Used to stratify results: identity measured over a local alignment's own region is
    biased upwards, because a local alignment keeps only the best-matching core.
    """
    pairs, _ = align(query, target, SCHEMES["blosum_semi"])
    matches = sum(query.sequence[i] == target.sequence[j] for i, j in pairs)
    return matches / max(1, min(len(query), len(target)))


def reference_alignment(usalign: str, path_query: str, path_target: str) -> dict | None:
    """Structural reference alignment from US-align.

    Returns the aligned residue pairs, the TM-scores keyed by normalization length, and the
    two ungapped sequences (so the caller can confirm US-align used the same residues).
    """
    result = subprocess.run([usalign, path_query, path_target],
                            capture_output=True, text=True)
    lines = result.stdout.splitlines()
    tm_by_length = {}
    for line in lines:
        if line.startswith("TM-score="):
            match = re.search(r"L=(\d+)", line)
            if match:
                tm_by_length[int(match.group(1))] = float(line.split("=")[1].split()[0])
    # The three alignment lines follow the '(":" denotes ...)' legend. The middle line can
    # start with spaces, so take them by position rather than by filtering.
    legend = next((i for i, line in enumerate(lines) if line.startswith('(":"')), None)
    if legend is None or not tm_by_length:
        return None
    body = [line for line in lines[legend + 1:] if line.strip()]
    if len(body) < 3:
        return None
    top, _, bottom = body[0], body[1], body[2]
    pairs, i, j = [], 0, 0
    for top_char, bottom_char in zip(top, bottom):
        if top_char != "-" and bottom_char != "-":
            pairs.append((i, j))
        i += top_char != "-"
        j += bottom_char != "-"
    return dict(pairs=np.array(pairs), tm_by_length=tm_by_length,
                sequence_query=top.replace("-", ""), sequence_target=bottom.replace("-", ""))


def reference_tm(reference: dict, length_norm: int) -> float:
    """US-align's own TM-score, normalized by the shorter chain where available."""
    return reference["tm_by_length"].get(length_norm, max(reference["tm_by_length"].values()))


def prf(predicted: np.ndarray, reference: np.ndarray, tolerance: int = 0) -> tuple[float, float, float]:
    """Precision, recall and F1 of predicted residue pairs against a reference alignment.

    ``tolerance`` allows a pair to count as correct if the target residue is within that
    many positions of the reference, which separates "aligned the wrong region" from
    "aligned the right region, shifted by a residue or two".
    """
    if len(predicted) == 0 or len(reference) == 0:
        return 0.0, 0.0, 0.0
    if tolerance == 0:
        reference_set = set(map(tuple, reference))
        hits = sum(tuple(pair) in reference_set for pair in predicted)
    else:
        by_query: dict[int, list[int]] = {}
        for i, j in reference:
            by_query.setdefault(i, []).append(j)
        hits = sum(any(abs(j - ref_j) <= tolerance for ref_j in by_query.get(i, []))
                   for i, j in predicted)
    precision, recall = hits / len(predicted), hits / len(reference)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _confusion_table(mode: str) -> np.ndarray:
    """Per-state distribution over the wrong 3Di states.

    ``confusion`` draws wrong states in proportion to their 3Di substitution score, so
    mistakes land on geometrically similar states, the way real predictors fail.
    ``uniform`` spreads them evenly, a pessimistic floor.
    """
    scores = SubstitutionMatrix.std_3di_matrix().score_matrix().astype(float)
    table = np.exp(scores) if mode == "confusion" else np.ones_like(scores)
    np.fill_diagonal(table, 0.0)
    return table / table.sum(axis=1, keepdims=True)


def degrade_3di(chain: Chain, accuracy: float, mode: str, rng: random.Random) -> Chain:
    """Replace a share of a chain's 3Di states, simulating an imperfect predictor."""
    if accuracy >= 1.0:
        return chain
    table = _confusion_table(mode)
    alphabet = "".join(SubstitutionMatrix.std_3di_matrix().get_alphabet1().get_symbols())
    states = list(chain.three_di)
    for position, state in enumerate(states):
        if state not in alphabet or rng.random() < accuracy:
            continue
        weights = table[alphabet.index(state)]
        states[position] = alphabet[rng.choices(range(len(alphabet)), weights=weights)[0]]
    # A profile would no longer match the degraded string.
    return replace(chain, three_di="".join(states), three_di_profile=None)


def write_ca_pdb(chain: Chain, path: str, chain_id: str = "A") -> None:
    """Write a CA-only PDB file, so US-align sees exactly the residues in ``chain``."""
    with open(path, "w") as handle:
        for number, (residue, xyz) in enumerate(zip(chain.sequence, chain.ca_coord), start=1):
            try:
                name = ProteinSequence.convert_letter_1to3(residue)
            except Exception:
                name = "UNK"
            handle.write(
                f"ATOM  {number:5d}  CA  {name:>3s} {chain_id}{number:4d}    "
                f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00           C\n"
            )
        handle.write("TER\nEND\n")
