from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

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


def _align_rows(args: tuple[int, int, list[str], list[str]]) -> list[tuple[str, str, float]]:
    """Worker function: align all pairs (i, j>i) for i in [i_start, i_end).

    Receives the full headers/seqs lists so it can generate its own pairs without
    any per-pair IPC overhead. seq_i is constructed once per row.
    """
    i_start, i_end, headers, seqs = args
    matrix = align.SubstitutionMatrix.std_protein_matrix()
    n = len(headers)
    results: list[tuple[str, str, float]] = []
    for i in range(i_start, i_end):
        try:
            seq_i = ProteinSequence(seqs[i])
        except Exception as exc:
            print(f"Warning: could not parse sequence {headers[i]}: {exc}")
            for j in range(i + 1, n):
                results.append((headers[i], headers[j], float("nan")))
            continue
        for j in range(i + 1, n):
            try:
                seq_j = ProteinSequence(seqs[j])
                alignments = align.align_optimal(
                    seq_i,
                    seq_j,
                    matrix,
                    gap_penalty=(-10, -1),
                    terminal_penalty=False,
                    local=False,
                )
                identity = align.get_sequence_identity(alignments[0])
            except Exception as exc:
                print(f"Warning: alignment failed for {headers[i]} vs {headers[j]}: {exc}")
                identity = float("nan")
            results.append((headers[i], headers[j], identity))
    return results


def _split_rows(n: int, workers: int) -> list[tuple[int, int]]:
    """Split row indices into *workers* chunks with approximately equal pair counts."""
    total = n * (n - 1) // 2
    target = total / workers
    chunks: list[tuple[int, int]] = []
    start = 0
    accumulated = 0
    for i in range(n):
        accumulated += n - 1 - i
        if accumulated >= target and len(chunks) < workers - 1:
            chunks.append((start, i + 1))
            start = i + 1
            accumulated = 0
    if start < n:
        chunks.append((start, n))
    return chunks


def run_pairwise_alignments(
    records: list[tuple[str, ProteinSequence]],
    workers: int,
) -> list[tuple[str, str, float]]:
    """Return (header1, header2, identity) for every unique pair using *workers* processes."""
    n = len(records)
    n_pairs = n * (n - 1) // 2
    headers = [r[0] for r in records]
    seqs = [str(r[1]) for r in records]

    row_chunks = _split_rows(n, workers)
    tasks = [(i_start, i_end, headers, seqs) for i_start, i_end in row_chunks]

    print(f"Starting {n_pairs} alignments with {workers} workers ({len(tasks)} tasks)")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        batch_results = list(tqdm(
            executor.map(_align_rows, tasks),
            total=len(tasks),
            desc="Aligning pairs",
        ))

    return [result for batch in batch_results for result in batch]


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
