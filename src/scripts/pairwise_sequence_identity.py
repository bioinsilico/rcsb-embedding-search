from __future__ import annotations

import argparse
import os
import shutil
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
            chain_id = atom_array.chain_id[start_idx] or "0"
            records.append((f"{stem}|{chain_id}", sequence))

    return records


_WRITE_BATCH_SIZE = 10_000


def _align_rows(args: tuple[int, int, list[str], list[str], str]) -> int:
    """Worker function: align all pairs (i, j>i) for i in [i_start, i_end).

    Writes results to *output_file* in batches of *_WRITE_BATCH_SIZE* lines.
    Returns the number of pairs written.
    """
    i_start, i_end, headers, seqs, output_file = args
    matrix = align.SubstitutionMatrix.std_protein_matrix()
    n = len(headers)
    count = 0
    batch: list[str] = []
    with open(output_file, "w") as f:
        for i in range(i_start, i_end):
            try:
                seq_i = ProteinSequence(seqs[i])
            except Exception as exc:
                print(f"Warning: could not parse sequence {headers[i]}: {exc}")
                for j in range(i + 1, n):
                    batch.append(f"{headers[i]}\t{headers[j]}\tnan\n")
                    count += 1
                    if len(batch) >= _WRITE_BATCH_SIZE:
                        f.writelines(batch)
                        batch.clear()
                continue
            for j in range(i + 1, n):
                try:
                    seq_j = ProteinSequence(seqs[j])
                    n_seq = min(len(seq_i), len(seq_j))
                    alignments = align.align_optimal(
                        seq_i,
                        seq_j,
                        matrix,
                        gap_penalty=(-10, -1),
                        terminal_penalty=False,
                        local=False,
                    )
                    trace = alignments[0].trace
                    if len(trace) == 0:
                        identity = 0.0
                    else:
                        matches = sum(
                            1
                            for pos1, pos2 in trace
                            if pos1 != -1 and pos2 != -1 and seq_i[pos1] == seq_j[pos2]
                        )
                        identity = matches / n_seq
                except Exception as exc:
                    raise RuntimeError(f"Alignment failed for {headers[i]} vs {headers[j]}") from exc
                batch.append(f"{headers[i]}\t{headers[j]}\t{identity:.4f}\n")
                count += 1
                if len(batch) >= _WRITE_BATCH_SIZE:
                    f.writelines(batch)
                    batch.clear()
        if batch:
            f.writelines(batch)
    return count


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


def window_split(
    records: list[tuple[str, ProteinSequence]],
    window_size: int,
    window_step: int,
) -> list[tuple[str, ProteinSequence]]:
    """Split each sequence into overlapping windows.

    Each output header is ``original_header|start-end`` (1-based, inclusive).
    Sequences shorter than *window_size* are emitted as a single window.
    """
    windowed: list[tuple[str, ProteinSequence]] = []
    for header, sequence in records:
        seq_str = str(sequence)
        seq_len = len(seq_str)
        if seq_len <= window_size:
            padded = seq_str + "X" * (window_size - seq_len)
            windowed.append((f"{header}|1-{seq_len}", ProteinSequence(padded)))
            continue
        for start in range(0, seq_len - window_size + 1, window_step):
            end = start + window_size
            windowed.append((
                f"{header}|{start + 1}-{end}",
                ProteinSequence(seq_str[start:end]),
            ))
        # include trailing fragment if the last window didn't reach the end
        if end < seq_len:
            windowed.append((
                f"{header}|{seq_len - window_size + 1}-{seq_len}",
                ProteinSequence(seq_str[seq_len - window_size:]),
            ))
    return windowed


def run_pairwise_alignments(
    records: list[tuple[str, ProteinSequence]],
    workers: int,
    output_file: str,
    tmp_dir: str | None = None,
) -> None:
    """Compute all pairwise alignments and write results to *output_file*.

    Each worker writes to its own temporary file to avoid concurrent writes.
    Temporary files are placed in *tmp_dir* (defaults to the directory of *output_file*).
    They are merged (with a TSV header) into *output_file* and then deleted.
    """
    n = len(records)
    n_pairs = n * (n - 1) // 2
    headers = [r[0] for r in records]
    seqs = [str(r[1]) for r in records]

    row_chunks = _split_rows(n, workers)
    base_name = os.path.splitext(os.path.basename(output_file))[0]
    ext = os.path.splitext(output_file)[1]
    work_dir = tmp_dir if tmp_dir is not None else os.path.dirname(os.path.abspath(output_file))
    worker_files = [os.path.join(work_dir, f"{base_name}.worker_{idx}{ext}") for idx in range(len(row_chunks))]
    tasks = [
        (i_start, i_end, headers, seqs, wf)
        for (i_start, i_end), wf in zip(row_chunks, worker_files)
    ]

    print(f"Starting {n_pairs} alignments with {workers} workers ({len(tasks)} tasks)")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        counts = list(tqdm(
            executor.map(_align_rows, tasks),
            total=len(tasks),
            desc="Aligning pairs",
        ))

    print(f"Merging {len(worker_files)} worker files into {output_file} ...")
    with open(output_file, "w") as out:
        out.write("seq1\tseq2\tsequence_identity\n")
        for wf in worker_files:
            with open(wf) as f:
                shutil.copyfileobj(f, out)
            os.remove(wf)

    print(f"Total pairs written: {sum(counts)}")


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
    parser.add_argument(
        "--tmp-dir", "-t", default=None,
        help=(
            "Directory for per-worker temporary files (default: same directory as --identity-output)."
        ),
    )
    parser.add_argument(
        "--window-size", "-W", type=int, default=None,
        help="Split sequences using a moving window of this size (residues). Disabled by default.",
    )
    parser.add_argument(
        "--window-step", "-S", type=int, default=None,
        help="Step size for the moving window (default: same as --window-size, i.e. non-overlapping).",
    )
    args = parser.parse_args()

    if args.window_step is not None and args.window_size is None:
        parser.error("--window-step requires --window-size")
    if args.window_size is not None and args.window_step is None:
        args.window_step = args.window_size

    print(f"Collecting sequences from {args.structure_path} ...")
    records = collect_sequences(args.structure_path, args.format, args.extension)
    print(f"Found {len(records)} sequences")

    if args.window_size is not None:
        records = window_split(records, args.window_size, args.window_step)
        print(f"After windowing ({args.window_size}/{args.window_step}): {len(records)} fragments")

    with open(args.fasta_output, "w") as fasta_file:
        for header, sequence in records:
            fasta_file.write(f">{header}\n{sequence}\n")
    print(f"Sequences written to {args.fasta_output}")

    n_pairs = len(records) * (len(records) - 1) // 2
    print(f"Computing pairwise alignments ({n_pairs} pairs) using {args.workers} workers ...")
    run_pairwise_alignments(records, args.workers, args.identity_output, args.tmp_dir)
    print(f"Pairwise identities written to {args.identity_output}")


if __name__ == "__main__":
    main()
