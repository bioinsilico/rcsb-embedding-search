"""
Pairwise alignment of two proteins scored with ESM3 per-residue embeddings.

  1. ESM3 (esm3_sm_open_v1) embedding for every residue of protein A and protein B
  2. cosine-similarity matrix, used in place of a substitution matrix such as BLOSUM62
  3. optional row/column z-score normalization of that matrix
  4. Needleman-Wunsch (global) or Smith-Waterman (local) alignment with affine gap
     penalties (Gotoh), selected with ALGORITHM
  5. EMBOSS-style printout, summary, and an optional BLOSUM62 alignment for comparison

Setup
  pip install esm httpx     # esm 3.2.1 imports httpx without declaring it as a dependency
  The weights are gated: accept the license for EvolutionaryScale/esm3-sm-open-v1 on
  Hugging Face, then log in once:  from huggingface_hub import login; login()
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial.distance import cdist

from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, LogitsConfig
from esm.utils.structure.protein_chain import ProteinChain

# =============================================================================
# Inputs
# =============================================================================
# Example pair: human hemoglobin alpha vs. beta (PDB 4HHB, entities 1 and 2). Replace with yours.
protein_A = (
    "MHHHHHHSSGVDLGTENLYFQSMAARRALHFVFKVGNRFQTARFYRDVLGMKVLRHEEFEEGCKAACNGPYDGKWSKTMVGFG"
    "PEDDHFVAELTYNYGVGDYKLGNDFMGITLASSQAVSNARKLEWPLTEVAEGVFETEAPGGYKFYLQNRSLPQSDPVLKVTLA"
    "VSDLQKSLNYWCNLLGMKIYENDEEKQRALLGYADNQCKLELQGVKGGVDHAAAFGRIAFSCPQKELPDLEDLMKRENQKILT"
    "PLVSLDTPGKATVQVVILADPDGHEICFVGDEAFRELSKMDPEGSNCWIDSKGGYGSEFELRRQACGRTRAPPPPPLRSGC"
)
protein_B = (
    "MHHHHHHSSGVDLGTENLYFQSNAMYEIKGHHHISMVTKNANENNHFYKNVLGLRRVKMTVNQDDPSMYHLFYGDKTGSPGTE"
    "LSFFEIPLVGRTYRGTNAITRIGLLVPSEDSLHYWKERFEKFDVKHSEMTTYANRPALQFEDAEGLRLVLLVSNGEKVEHWET"
    "WEKSEVPAKHQIQGMGSVELTVRRLDKMASTLTEIFGYTEVSRNDQEAIFQSIKGEAFGEIVVKYLDGPTEKPGRGSIHHLAI"
    "RVKNDAELAYWEEQVKQRGFHSSGIIDRFYFKSLYFRESNGILFEIATDGPGFTVDGDVEHLGEKLDLPPFLEDQRAEIEANL"
    "APIEEK"
)

# Optional structure conditioning: a PDB-format file path or a 4-character PDB ID.
# When set, the sequence AND backbone coordinates are taken from that chain
# (modelled residues only) and protein_A / protein_B are ignored.
STRUCTURE_A, CHAIN_A = None, "A"  # e.g. "4HHB", "A"   or   "my_model.pdb", "A"
STRUCTURE_B, CHAIN_B = None, "B"  # e.g. "4HHB", "B"

# =============================================================================
# Scoring and alignment parameters
# =============================================================================
# "needleman-wunsch": global, aligns the full length of both proteins
# "smith-waterman"  : local, only the best-matching region (e.g. a shared domain)
ALGORITHM = "needleman-wunsch"

# "rowcol": mean of row-wise and column-wise z-scores ("signal enhancement" from EBA,
#           Pantolini et al. 2024, Bioinformatics 40:btad786). A pair scores high only if
#           it stands out within its row AND its column.
# "zscore": a single z-score over the whole matrix.
# "none"  : raw cosine similarity (rescale the gap penalties and offset to the cosine range).
NORMALIZATION = "rowcol"
GAP_OPEN = 5.0         # penalty for the first position of a gap (z-score units)
GAP_EXTEND = 0.5       # each further position: a gap of length L costs GAP_OPEN + (L-1)*GAP_EXTEND

# Needleman-Wunsch only
FREE_END_GAPS = True  # True: N/C-terminal overhangs are free (e.g. different construct boundaries)

# Smith-Waterman only: subtracted from every pair score so that unrelated pairs score below
# zero (EBA subtracted 2 from its row/column z-scores). Lower values give longer local
# alignments; higher values keep only the strongest region. Same units as the score matrix.
LOCAL_SCORE_OFFSET = 2.0

COMPARE_WITH_BLOSUM62 = True  # also align with BLOSUM62 (gap 10/0.5) using the same algorithm


# =============================================================================
# ESM3 embeddings
# =============================================================================
def load_protein(sequence: str | None, structure: str | None = None, chain_id: str = "detect") -> ESMProtein:
    """ESMProtein from a sequence, or from a chain of a PDB file / PDB ID (sequence + coordinates)."""
    if structure is None:
        return ESMProtein(sequence="".join(sequence.split()).upper())  # tolerate pasted line breaks
    if len(structure) == 4 and not os.path.exists(structure):
        chain = ProteinChain.from_rcsb(structure, chain_id)  # downloaded from RCSB PDB
    else:
        chain = ProteinChain.from_pdb(structure, chain_id)
    return ESMProtein.from_protein_chain(chain)


def residue_embeddings(model: ESM3InferenceClient, protein: ESMProtein) -> np.ndarray:
    """Per-residue ESM3 embeddings, shape [L, 1536].

    These are the output of the last transformer block (before the final LayerNorm).
    ESM3 sums all input tracks (sequence, structure, ...) into one embedding before the
    first block, so every layer works on the fused representation.
    """
    with torch.no_grad():
        protein_tensor = model.encode(protein)  # tokenizes sequence (+ structure if coordinates are set)
        output = model.logits(protein_tensor, LogitsConfig(return_embeddings=True))
    emb = output.embeddings[0, 1:-1]  # [L + 2, d] -> drop the BOS and EOS positions
    if emb.shape[0] != len(protein.sequence):
        raise RuntimeError(f"expected {len(protein.sequence)} residue embeddings, got {emb.shape[0]}")
    return emb.float().cpu().numpy()


# =============================================================================
# Score matrix
# =============================================================================
def normalize_scores(sim: np.ndarray, mode: str = "rowcol") -> np.ndarray:
    sim = np.asarray(sim, dtype=np.float64)
    eps = 1e-8
    if mode == "none":
        return sim
    if mode == "zscore":
        return (sim - sim.mean()) / (sim.std() + eps)
    if mode == "rowcol":
        z_row = (sim - sim.mean(axis=1, keepdims=True)) / (sim.std(axis=1, keepdims=True) + eps)
        z_col = (sim - sim.mean(axis=0, keepdims=True)) / (sim.std(axis=0, keepdims=True) + eps)
        return 0.5 * (z_row + z_col)
    raise ValueError(f"unknown normalization mode: {mode!r}")


def blosum62_scores(seq_a: str, seq_b: str) -> np.ndarray:
    """BLOSUM62 looked up for every residue pair, so it can go through the same aligner."""
    from Bio.Align import substitution_matrices  # Biopython is installed with esm

    b62 = substitution_matrices.load("BLOSUM62")
    index = {aa: k for k, aa in enumerate(b62.alphabet)}
    ia = [index.get(aa, index["X"]) for aa in seq_a]
    ib = [index.get(aa, index["X"]) for aa in seq_b]
    return np.asarray(b62)[np.ix_(ia, ib)]


# =============================================================================
# Needleman-Wunsch (global) and Smith-Waterman (local), affine gaps
# =============================================================================
@dataclass
class Alignment:
    score: float
    pairs: list[tuple[int | None, int | None]]  # (index in A, index in B), None = gap

    def gapped(self, seq_a: str, seq_b: str) -> tuple[str, str]:
        top = "".join("-" if i is None else seq_a[i] for i, _ in self.pairs)
        bottom = "".join("-" if j is None else seq_b[j] for _, j in self.pairs)
        return top, bottom

    def aligned_pairs(self) -> list[tuple[int, int]]:
        return [(i, j) for i, j in self.pairs if i is not None and j is not None]


def needleman_wunsch(
    scores: np.ndarray,
    gap_open: float,
    gap_extend: float | None = None,
    free_end_gaps: bool = False,
) -> Alignment:
    """Global alignment of the full sequences (Needleman & Wunsch 1970).

    scores[i, j]   score for aligning residue i of A with residue j of B (shape n x m)
    gap_open       penalty (positive) for the first position of a gap
    gap_extend     penalty for each further position; None -> gap_open (linear gaps)
    free_end_gaps  leading/trailing gaps cost nothing (semi-global)
    """
    return _align(scores, gap_open, gap_extend, local=False, free_end_gaps=free_end_gaps)


def smith_waterman(
    scores: np.ndarray,
    gap_open: float,
    gap_extend: float | None = None,
) -> Alignment:
    """Local alignment: the highest-scoring pair of segments (Smith & Waterman 1981).

    Only meaningful when unrelated residue pairs score below zero on average, so shift
    z-scored matrices down first (LOCAL_SCORE_OFFSET). Returns an empty alignment
    (no pairs, score 0) when no residue pair scores above zero.
    """
    return _align(scores, gap_open, gap_extend, local=True, free_end_gaps=False)


def _align(scores, gap_open, gap_extend, local: bool, free_end_gaps: bool) -> Alignment:
    """Dynamic programming shared by both algorithms (Gotoh 1982 affine-gap recurrences).

    States: M = residue pair, X = A residue vs gap, Y = gap vs B residue, H = max(M, X, Y);
    Smith-Waterman adds 0 as a fourth option ("start a new alignment here").
    A gap of length L costs gap_open + (L - 1) * gap_extend.
    Each DP row is filled with numpy: M and X need only the previous row, and the
    within-row Y recurrence is solved with a running maximum, so Python loops over rows only.
    """
    S = np.asarray(scores, dtype=np.float64)
    if S.ndim != 2:
        raise ValueError("scores must be a 2-D matrix")
    n, m = S.shape
    go = float(gap_open)
    ge = go if gap_extend is None else float(gap_extend)
    if not go >= ge >= 0:
        raise ValueError("need gap_open >= gap_extend >= 0")

    cols = np.arange(m + 1)

    def edge_score(k):  # first row / column: a leading gap of length k
        if local or free_end_gaps:
            return np.zeros_like(k, dtype=np.float64)
        return np.where(k > 0, -(go + (k - 1) * ge), 0.0)

    # Traceback flags
    h_from_y = np.zeros((n + 1, m + 1), dtype=bool)  # H[i,j] = Y[i,j]  (else V = max(M, X[, 0]))
    v_from_x = np.zeros((n + 1, m + 1), dtype=bool)  # V[i,j] = X[i,j]  (else M, diagonal)
    x_extend = np.zeros((n + 1, m + 1), dtype=bool)  # X[i,j] extends X[i-1,j] (else opens from H[i-1,j])
    y_extend = np.zeros((n + 1, m + 1), dtype=bool)  # Y[i,j] extends Y[i,j-1] (else opens from V[i,j-1])
    start = np.zeros((n + 1, m + 1), dtype=bool)     # the alignment starts here: traceback stops

    # Row 0
    H_prev = edge_score(cols)
    X_prev = np.full(m + 1, -np.inf)
    last_col = np.empty(n + 1)
    last_col[0] = H_prev[m]
    if local:
        start[0, :] = True
    else:
        start[0, 0] = True
        h_from_y[0, 1:] = True  # leading gap in A
        y_extend[0, 2:] = True
    best_score, best_i, best_j = 0.0, 0, 0  # Smith-Waterman: best cell so far

    for i in range(1, n + 1):
        # X: A[i-1] against a gap (vertical move)
        x_open = H_prev - go
        x_ext = X_prev - ge
        X = np.maximum(x_open, x_ext)
        x_extend[i] = x_ext > x_open

        # M: A[i-1] paired with B[j-1] (diagonal move)
        M = np.empty(m + 1)
        M[0] = -np.inf
        M[1:] = H_prev[:-1] + S[i - 1]

        V = np.maximum(M, X)
        v_from_x[i] = X > M
        if local:
            # Smith-Waterman: a prefix scoring <= 0 is dropped and a new alignment starts
            start[i] = V <= 0
            V = np.maximum(V, 0.0)
            X[0] = -np.inf
        else:
            # Column 0: leading gap in B
            V[0] = X[0] = edge_score(np.array(i))
            v_from_x[i, 0] = True
            x_extend[i, 0] = i > 1

        # Y: gap against B[j-1] (horizontal move)
        #   Y[j] = max(V[j-1] - go, Y[j-1] - ge) = max_{k<j}(V[k] + ge*k) - go - ge*(j-1)
        #   (opening from V rather than H is exact because gap_open >= gap_extend)
        Y = np.empty(m + 1)
        Y[0] = -np.inf
        if m:
            Y[1:] = np.maximum.accumulate(V[:-1] + ge * cols[:-1]) - go - ge * cols[:-1]
            y_extend[i, 1:] = (Y[:-1] - ge) > (V[:-1] - go)

        H = np.maximum(V, Y)
        h_from_y[i] = Y > V

        if local:
            j_max = int(np.argmax(H))
            if H[j_max] > best_score:
                best_score, best_i, best_j = float(H[j_max]), i, j_max
        H_prev, X_prev = H, X
        last_col[i] = H[m]

    # Where the traceback starts
    rev: list[tuple[int | None, int | None]] = []  # built in reverse
    if local:
        i, j, score = best_i, best_j, best_score
    elif free_end_gaps:
        j_end = m - int(np.argmax(H_prev[::-1]))  # best cell in the last row (ties -> j = m)
        i_end = n - int(np.argmax(last_col[::-1]))  # best cell in the last column (ties -> i = n)
        if H_prev[j_end] >= last_col[i_end]:
            i, j, score = n, j_end, float(H_prev[j_end])
        else:
            i, j, score = i_end, m, float(last_col[i_end])
        rev.extend((None, jj) for jj in range(m - 1, j - 1, -1))  # free trailing gap in A
        rev.extend((ii, None) for ii in range(n - 1, i - 1, -1))  # free trailing gap in B
    else:
        i, j, score = n, m, float(H_prev[m])

    # Traceback; ties prefer the diagonal
    state = "H"
    while True:
        if state == "H":
            state = "Y" if h_from_y[i, j] else "V"
        elif state == "V":
            if start[i, j]:
                break
            state = "X" if v_from_x[i, j] else "M"
        elif state == "M":
            rev.append((i - 1, j - 1))
            i, j, state = i - 1, j - 1, "H"
        elif state == "X":
            rev.append((i - 1, None))
            state = "X" if x_extend[i, j] else "H"
            i -= 1
        else:  # "Y"
            rev.append((None, j - 1))
            state = "Y" if y_extend[i, j] else "V"
            j -= 1
    rev.reverse()
    return Alignment(score=score, pairs=rev)


# =============================================================================
# Output
# =============================================================================
def format_alignment(aln: Alignment, seq_a: str, seq_b: str, scores: np.ndarray,
                     name_a: str = "A", name_b: str = "B", width: int = 60) -> str:
    """EMBOSS-style blocks with residue numbers: '|' identical, ':' other pair in the top 5% of scores."""
    top, bottom = aln.gapped(seq_a, seq_b)
    strong = np.quantile(scores, 0.95)
    mid = "".join(
        " " if i is None or j is None
        else "|" if seq_a[i] == seq_b[j]
        else ":" if scores[i, j] >= strong
        else " "
        for i, j in aln.pairs
    )
    w = max(len(name_a), len(name_b))
    lines, end_a, end_b = [], 0, 0
    for s in range(0, len(aln.pairs), width):
        block = aln.pairs[s:s + width]
        ia = [i for i, _ in block if i is not None]
        ib = [j for _, j in block if j is not None]
        start_a, end_a = (ia[0] + 1, ia[-1] + 1) if ia else (end_a, end_a)
        start_b, end_b = (ib[0] + 1, ib[-1] + 1) if ib else (end_b, end_b)
        lines += [
            f"{name_a:<{w}} {start_a:>5} {top[s:s + width]} {end_a}",
            f"{'':<{w}} {'':>5} {mid[s:s + width]}",
            f"{name_b:<{w}} {start_b:>5} {bottom[s:s + width]} {end_b}",
            "",
        ]
    return "\n".join(lines)


def summary(aln: Alignment, seq_a: str, seq_b: str, similarity: np.ndarray | None = None) -> str:
    pairs = aln.aligned_pairs()
    if not pairs:
        return f"No aligned residues (score {aln.score:.2f})"
    L = len(aln.pairs)
    ident = sum(seq_a[i] == seq_b[j] for i, j in pairs)
    n_aligned = len(pairs)  # columns with a residue from both A and B
    gaps = L - n_aligned
    (first_a, first_b), (last_a, last_b) = pairs[0], pairs[-1]
    out = [
        f"Length:   {L}  (sequence lengths A: {len(seq_a)}, B: {len(seq_b)})",
        f"Region:   A {first_a + 1}-{last_a + 1}, B {first_b + 1}-{last_b + 1}",
        f"Aligned:  {n_aligned} residue pairs ({100 * n_aligned / len(seq_a):.1f}% of A, "
        f"{100 * n_aligned / len(seq_b):.1f}% of B)",
        f"Identity: {ident}/{L} ({100 * ident / L:.1f}%)",
        f"Gaps:     {gaps}/{L} ({100 * gaps / L:.1f}%)",
        f"Score:    {aln.score:.2f}",
    ]
    if similarity is not None:
        idx = np.array(pairs)
        mean_cos = similarity[idx[:, 0], idx[:, 1]].mean()
        out.append(f"Mean ESM3 cosine similarity of aligned pairs: {mean_cos:.3f}")
    return "\n".join(out)


# =============================================================================
# Run
# =============================================================================
def main() -> None:
    algorithm = ALGORITHM.strip().lower().replace("_", "-").replace(" ", "-")
    if algorithm not in ("needleman-wunsch", "smith-waterman"):
        raise ValueError(f'ALGORITHM must be "needleman-wunsch" or "smith-waterman", not {ALGORITHM!r}')
    local = algorithm == "smith-waterman"
    method = "Smith-Waterman (local)" if local else "Needleman-Wunsch (global)"

    def align(scores: np.ndarray, gap_open: float, gap_extend: float, offset: float = 0.0) -> Alignment:
        if local:
            return smith_waterman(scores - offset, gap_open, gap_extend)
        return needleman_wunsch(scores, gap_open, gap_extend, free_end_gaps=FREE_END_GAPS)

    # 1. Load the open-weights model (GPU if available, otherwise CPU)
    model: ESM3InferenceClient = ESM3.from_pretrained("esm3_sm_open_v1")

    # 2. Build ESMProtein inputs (sequence only, or sequence + coordinates)
    input_A = load_protein(protein_A, STRUCTURE_A, CHAIN_A)
    input_B = load_protein(protein_B, STRUCTURE_B, CHAIN_B)
    seq_A, seq_B = input_A.sequence, input_B.sequence
    name_A = f"{STRUCTURE_A}:{CHAIN_A}" if STRUCTURE_A else "A"
    name_B = f"{STRUCTURE_B}:{CHAIN_B}" if STRUCTURE_B else "B"

    # 3. Per-residue embeddings, shape [L, 1536]
    embeddings_A = residue_embeddings(model, input_A)
    embeddings_B = residue_embeddings(model, input_B)

    # 4. Cosine similarity matrix [L_A, L_B] -> alignment score matrix
    similarity_matrix = 1 - cdist(embeddings_A, embeddings_B, metric="cosine")
    score_matrix = normalize_scores(similarity_matrix, NORMALIZATION)

    # 5. Pairwise alignment (Needleman-Wunsch or Smith-Waterman, see ALGORITHM)
    aln = align(score_matrix, GAP_OPEN, GAP_EXTEND, offset=LOCAL_SCORE_OFFSET)

    setting = f"score offset {LOCAL_SCORE_OFFSET}" if local else f"free_end_gaps={FREE_END_GAPS}"
    print(f"# ESM3 embedding alignment, {method}  "
          f"(normalization={NORMALIZATION}, gap {GAP_OPEN}/{GAP_EXTEND}, {setting})")
    if aln.pairs:
        print(summary(aln, seq_A, seq_B, similarity_matrix))
        print("# '|' identical, ':' non-identical pair in the top 5% of scores\n")
        print(format_alignment(aln, seq_A, seq_B, score_matrix, name_A, name_B))
    else:
        print("No residue pair scores above zero: lower LOCAL_SCORE_OFFSET (or check NORMALIZATION).\n")

    if COMPARE_WITH_BLOSUM62:
        blosum = blosum62_scores(seq_A, seq_B)
        ref = align(blosum, 10.0, 0.5)  # BLOSUM62 already scores unrelated pairs below zero: no offset
        esm_pairs, ref_pairs = set(aln.aligned_pairs()), set(ref.aligned_pairs())
        shared = len(esm_pairs & ref_pairs)
        print(f"# BLOSUM62 alignment, {method}  (gap 10.0/0.5), same aligner")
        print(summary(ref, seq_A, seq_B, similarity_matrix))
        if esm_pairs and ref_pairs:
            print(f"Aligned pairs shared with the ESM3 alignment: {shared}/{len(esm_pairs)} "
                  f"({100 * shared / len(esm_pairs):.1f}%)")
        print()
        if ref.pairs:
            print(format_alignment(ref, seq_A, seq_B, blosum, name_A, name_B))


if __name__ == "__main__":
    main()
