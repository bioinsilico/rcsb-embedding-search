from __future__ import annotations

import argparse
import logging

import numpy as np
from scipy.spatial.distance import cdist
from biotite.structure import filter_amino_acids, chain_iter, get_chains, AtomArrayStack
from biotite.structure.io.pdb import PDBFile, get_structure as get_pdb_structure
from biotite.structure.io.pdbx import CIFFile, get_structure as get_cif_structure

logger = logging.getLogger(__name__)


def read_structure(input_file: str, fmt: str):
    """Read a structure file and return a single-model AtomArray."""
    if fmt == "pdb":
        structure = get_pdb_structure(PDBFile.read(input_file))
        if isinstance(structure, AtomArrayStack):
            structure = structure[0]
        return structure
    elif fmt == "cif":
        return get_cif_structure(CIFFile.read(input_file), model=1, use_author_fields=False)
    else:
        raise ValueError(f"Unknown format: {fmt!r}. Use 'pdb' or 'cif'.")


def extract_ca_coords(atom_array) -> tuple[np.ndarray, list[str]]:
    """Return Ca coordinates (shape N×3) and residue labels for all amino acid residues.

    Labels have the form ``"{chain_id}:{res_id}"``. For CIF structures parsed with
    ``use_author_fields=False`` these reflect canonical (non-author) numbering.
    """
    aa_atoms = atom_array[filter_amino_acids(atom_array)]
    ca_atoms = aa_atoms[aa_atoms.atom_name == "CA"]
    labels = [f"{ch}:{rid}" for ch, rid in zip(ca_atoms.chain_id, ca_atoms.res_id)]
    return ca_atoms.coord, labels


def compute_neighbor_lists(ca_coords: np.ndarray, threshold: float) -> list[list[int]]:
    """
    For each residue, return the 0-based indices of residues whose Ca is
    within *threshold* Ångströms (self excluded).
    """
    dist_matrix = cdist(ca_coords, ca_coords)
    idx = np.arange(len(ca_coords))
    return [
        list(np.where((dist_matrix[i] <= threshold) & (idx != i))[0])
        for i in range(len(ca_coords))
    ]


def _fmt(i: int, res_labels: list[str] | None) -> str:
    """Format a residue token as ``{idx}`` or ``{idx}:{chain}:{res_num}``."""
    return f"{i}:{res_labels[i]}" if res_labels is not None else str(i)


def write_tsv(
    neighbors: list[list[int]],
    out_path: str,
    res_labels: list[str] | None = None,
) -> None:
    """Write neighbor lists to a TSV file.

    Each row: <residue_token>\\t<neighbor_token_0>\\t<neighbor_token_1>\\t...

    When *res_labels* is provided each token has the form ``{idx}:{chain_id}:{res_num}``;
    otherwise tokens are plain 0-based integers.
    The number of columns varies per row depending on neighbor count.
    """
    with open(out_path, "w") as f:
        for i, nbrs in enumerate(neighbors):
            row = "\t".join([_fmt(i, res_labels)] + [_fmt(j, res_labels) for j in nbrs])
            f.write(row + "\n")
    logger.info(f"Written {len(neighbors)} residues → {out_path}")


def ca_neighbors(
    input_file: str,
    out_path: str,
    fmt: str = "cif",
    level: str = "chain",
    threshold: float = 8.0,
    labels: bool = False,
) -> None:
    """
    Compute Ca-based residue neighbor lists for a protein structure file.

    Parameters
    ----------
    input_file : str
        Path to the input structure file (PDB or CIF).
    out_path : str
        Output path prefix.
        - Chain level  → one file per chain: ``{out_path}_{chain_id}.tsv``
        - Assembly level → single file: ``{out_path}.tsv``
    fmt : str
        Input file format: ``"pdb"`` or ``"cif"`` (default ``"cif"``).
    level : str
        ``"chain"`` processes each chain independently (residue indices restart
        at 0 for every chain); ``"assembly"`` treats all chains as one unit with
        global 0-based indices spanning the whole assembly.
    threshold : float
        Ca–Ca distance threshold in Ångströms (default ``8.0``).
    labels : bool
        When ``True`` each token in the TSV is formatted as
        ``{idx}:{chain_id}:{res_num}`` instead of a plain integer index.
        For CIF files the residue numbers are canonical (non-author) IDs.
    """
    structure = read_structure(input_file, fmt)

    if level == "chain":
        for atom_ch in chain_iter(structure):
            chain_id = get_chains(atom_ch)[0]
            ca_coords, res_labels = extract_ca_coords(atom_ch)
            if len(ca_coords) == 0:
                logger.warning(f"Chain {chain_id}: no Ca atoms found, skipping")
                continue
            neighbors = compute_neighbor_lists(ca_coords, threshold)
            write_tsv(neighbors, f"{out_path}_{chain_id}.tsv", res_labels if labels else None)

    elif level == "assembly":
        ca_coords, res_labels = extract_ca_coords(structure)
        if len(ca_coords) == 0:
            logger.warning("Assembly: no Ca atoms found")
            return
        neighbors = compute_neighbor_lists(ca_coords, threshold)
        write_tsv(neighbors, f"{out_path}.tsv", res_labels if labels else None)

    else:
        raise ValueError(f"Unknown level: {level!r}. Use 'chain' or 'assembly'.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute Ca-based residue neighbor lists from a protein structure file. "
            "Outputs a TSV where the first column is the 0-based residue index and "
            "subsequent columns list neighbor indices within the Ca distance threshold. "
            "The number of columns varies per row."
        )
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Input structure file (PDB or CIF format)"
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help=(
            "Output path prefix. "
            "Chain level: {prefix}_{chain_id}.tsv per chain; "
            "assembly level: {prefix}.tsv"
        )
    )
    parser.add_argument(
        "--format", "-f", choices=["pdb", "cif"], default="cif",
        help="Input file format (default: cif)"
    )
    parser.add_argument(
        "--level", choices=["chain", "assembly"], default="chain",
        help=(
            "Calculation level: 'chain' processes each chain independently with "
            "per-chain 0-based indices; 'assembly' uses global indices across all "
            "chains (default: chain)"
        )
    )
    parser.add_argument(
        "--threshold", type=float, default=8.0,
        help="Ca–Ca distance threshold in Ångströms (default: 8.0)"
    )
    parser.add_argument(
        "--labels", action="store_true",
        help=(
            "Annotate each index with chain ID and residue number as "
            "{idx}:{chain_id}:{res_num}. For CIF files the residue numbers "
            "are canonical (non-author) IDs."
        )
    )
    args = parser.parse_args()

    ca_neighbors(
        input_file=args.input,
        out_path=args.output,
        fmt=args.format,
        level=args.level,
        threshold=args.threshold,
        labels=args.labels,
    )


if __name__ == "__main__":
    import config.logging_config
    main()
