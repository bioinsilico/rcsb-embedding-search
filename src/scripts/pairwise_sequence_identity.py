from __future__ import annotations

import argparse
import itertools
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import biotite.sequence.align as align
import biotite.structure.io.pdb as pdb
import biotite.structure.io.pdbx as pdbx
from biotite.sequence import ProteinSequence
from biotite.structure import BadStructureError, filter_amino_acids, filter_polymer, to_sequence
from biotite.structure.atoms import AtomArrayStack
from tqdm import tqdm


def collect_sequences(input_dir: str, fmt: str, ext: str | None = None) -> list[tuple[str, ProteinSequence]]:
    """Return (header, sequence) pairs for all chains in structure files under *input_dir*.

    *ext* overrides the file extension filter. Pass an empty string to match all files.
    """
    if ext is None:
        ext = f".{fmt}"
    records: list[tuple[str, ProteinSequence]] = []

    for filename in sorted(os.listdir(input_dir)):
        if ext and not filename.endswith(ext):
            continue
        file_path = os.path.join(input_dir, filename)
        stem = filename[: -len(ext)] if ext else filename

        try:
            if fmt == "pdb":
                atom_array = pdb.PDBFile.read(file_path).get_structure()
                if isinstance(atom_array, AtomArrayStack):
                    atom_array = atom_array[0]
            else:
                atom_array = pdbx.get_structure(pdbx.CIFFile.read(file_path), model=1)
        except BadStructureError:
            print(f"Warning: could not read {filename}, skipping")
            continue

        atom_array = atom_array[filter_polymer(atom_array)]
        atom_array = atom_array[filter_amino_acids(atom_array)]

        try:
            sequences, chain_starts = to_sequence(atom_array)
        except BadStructureError:
            print(f"Warning: could not extract sequences from {filename}, skipping")
            continue

        for sequence, start_idx in zip(sequences, chain_starts):
            chain_id = atom_array.chain_id[start_idx]
            records.append((f"{stem}|{chain_id}", sequence))

    return records


def _align_pair(args: tuple[str, str, str, str]) -> tuple[str, str, float]:
    """Worker function: align a single pair and return (header_i, header_j, identity).

    Sequences are passed as strings and reconstructed inside the worker so the
    task is fully picklable across process boundaries.
    """
    header_i, seq_str_i, header_j, seq_str_j = args
    matrix = align.SubstitutionMatrix.std_protein_matrix()
    try:
        seq_i = ProteinSequence(seq_str_i)
        seq_j = ProteinSequence(seq_str_j)
        alignments = align.align_optimal(
            seq_i,
            seq_j,
            matrix,
            gap_penalty=(-10, -1),
            terminal_penalty=False,
        )
        best = alignments[0]
        trace = best.trace
        if len(trace) == 0:
            return header_i, header_j, 0.0
        matches = sum(
            1
            for pos1, pos2 in trace
            if pos1 != -1 and pos2 != -1 and seq_i[pos1] == seq_j[pos2]
        )
        identity = matches / len(trace)
    except Exception as exc:
        print(f"Warning: alignment failed for {header_i} vs {header_j}: {exc}")
        identity = float("nan")
    return header_i, header_j, identity


def run_pairwise_alignments(
    records: list[tuple[str, ProteinSequence]],
    workers: int,
) -> list[tuple[str, str, float]]:
    """Return (header1, header2, identity) for every unique pair using *workers* processes."""
    tasks = [
        (records[i][0], str(records[i][1]), records[j][0], str(records[j][1]))
        for i, j in itertools.combinations(range(len(records)), 2)
    ]

    print(f"Starting {len(tasks)} alignments with {workers} workers")
    results: list[tuple[str, str, float]] = []
    chunksize = max(1, len(tasks) // (workers * 4))

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_align_pair, task): task for task in tasks}
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Aligning pairs"):
            results.append(future.result())

    # Sort to get a deterministic output order (same as combinations order)
    order = {(records[i][0], records[j][0]): k for k, (i, j) in enumerate(itertools.combinations(range(len(records)), 2))}
    results.sort(key=lambda r: order.get((r[0], r[1]), order.get((r[1], r[0]), 0)))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Collect sequences from structure files in a directory, write them as FASTA, "
            "and compute pairwise global sequence identities."
        )
    )
    parser.add_argument(
        "--structure-path", "-s", required=True,
        help="Directory containing structure files",
    )
    parser.add_argument(
        "--format", "-f", choices=["pdb", "cif"], default="cif",
        help="Structure file format (default: cif)",
    )
    parser.add_argument(
        "--fasta-output", "-a", required=True,
        help="Output FASTA file for all collected sequences",
    )
    parser.add_argument(
        "--identity-output", "-i", required=True,
        help="Output TSV file for pairwise sequence identities",
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=os.cpu_count(),
        help=f"Number of parallel worker processes (default: {os.cpu_count()})",
    )
    parser.add_argument(
        "--extension", "-e", default=None,
        help=(
            "File extension filter, including the leading dot (e.g. '.ent', '.pdb.gz'). "
            "Pass an empty string to match all files. "
            "Defaults to the extension implied by --format (.pdb or .cif)."
        ),
    )
    args = parser.parse_args()

    print(f"Collecting sequences from {args.structure_path} ...")
    records = collect_sequences(args.structure_path, args.format, args.extension)
    print(f"Found {len(records)} sequences")

    with open(args.fasta_output, "w") as fasta_file:
        for header, sequence in records:
            fasta_file.write(f">{header}\n{sequence}\n")
    print(f"Sequences written to {args.fasta_output}")

    n_pairs = len(records) * (len(records) - 1) // 2
    print(f"Computing pairwise alignments ({n_pairs} pairs) using {args.workers} workers ...")
    results = run_pairwise_alignments(records, args.workers)

    with open(args.identity_output, "w") as tsv_file:
        tsv_file.write("seq1\tseq2\tsequence_identity\n")
        for header_i, header_j, identity in results:
            tsv_file.write(f"{header_i}\t{header_j}\t{identity:.4f}\n")
    print(f"Pairwise identities written to {args.identity_output}")


if __name__ == "__main__":
    main()
