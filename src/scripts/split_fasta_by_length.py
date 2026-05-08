from __future__ import annotations

import argparse
import os


def split_fasta_by_length(fasta_path: str, threshold: int) -> None:
    """Split a FASTA file into two files based on sequence length.

    Sequences with length <= *threshold* go to the ``-le<threshold>`` file;
    sequences with length > *threshold* go to the ``-gt<threshold>`` file.
    """
    base, ext = os.path.splitext(fasta_path)
    le_path = f"{base}-lt{threshold}{ext}"
    gt_path = f"{base}-gt{threshold}{ext}"

    le_count = 0
    gt_count = 0

    with open(fasta_path) as f_in, \
         open(le_path, "w") as f_le, \
         open(gt_path, "w") as f_gt:

        header = None
        seq_lines: list[str] = []

        def flush(header: str, seq_lines: list[str]) -> None:
            nonlocal le_count, gt_count
            seq = "".join(seq_lines)
            if len(seq) <= threshold:
                f_le.write(f">{header}\n{seq}\n")
                le_count += 1
            else:
                f_gt.write(f">{header}\n{seq}\n")
                gt_count += 1

        for line in f_in:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if header is not None:
                    flush(header, seq_lines)
                header = line[1:]
                seq_lines = []
            else:
                seq_lines.append(line)

        if header is not None:
            flush(header, seq_lines)

    print(f"Sequences <= {threshold}: {le_count:,}  -> {le_path}")
    print(f"Sequences >  {threshold}: {gt_count:,}  -> {gt_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Split a FASTA file into two files based on a sequence length threshold. "
            "Sequences with length <= threshold go to <name>-le<threshold><ext>; "
            "sequences with length > threshold go to <name>-gt<threshold><ext>."
        )
    )
    parser.add_argument(
        "--fasta-input", "-f", required=True,
        help="Path to the input FASTA file",
    )
    parser.add_argument(
        "--threshold", "-t", type=int, required=True,
        help="Sequence length threshold (inclusive upper bound for the 'le' file)",
    )
    args = parser.parse_args()
    split_fasta_by_length(args.fasta_input, args.threshold)


if __name__ == "__main__":
    main()
