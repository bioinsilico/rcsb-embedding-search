from __future__ import annotations

import argparse
import glob
import logging
import os

from scripts.ca_neighbors import ca_neighbors

logger = logging.getLogger(__name__)


def _is_ready(out_prefix: str, level: str) -> bool:
    """Check whether output already exists for a given domain."""
    if level == "assembly":
        return os.path.isfile(f"{out_prefix}.tsv")
    # chain level: at least one {prefix}_{chain}.tsv must exist
    return len(glob.glob(f"{out_prefix}_*.tsv")) > 0


def process_directory(
    pdb_path: str,
    out_path: str,
    fmt: str = "cif",
    level: str = "chain",
    threshold: float = 8.0,
    labels: bool = False,
    esm3_offsets: bool = False,
) -> None:
    """Compute Ca neighbors for every structure file in a directory.

    Output TSV files are written to *out_path* with the same base name as the
    input structure file.  Files that already have output are skipped, making
    it safe to resume an interrupted run.

    Parameters
    ----------
    pdb_path : str
        Directory containing structure files.
    out_path : str
        Output directory for neighbor TSV files.
    fmt : str
        Input file format: ``"pdb"`` or ``"cif"``.
    level, threshold, labels, esm3_offsets :
        Forwarded to :func:`scripts.ca_neighbors.ca_neighbors`.
    """
    os.makedirs(out_path, exist_ok=True)

    for filename in sorted(os.listdir(pdb_path)):
        input_file = os.path.join(pdb_path, filename)
        if not os.path.isfile(input_file):
            continue

        domain_id = os.path.splitext(filename)[0]
        out_prefix = os.path.join(out_path, domain_id)

        if _is_ready(out_prefix, level):
            logger.info(f"{domain_id} is ready")
            continue

        logger.info(f"Processing {filename}")
        try:
            ca_neighbors(
                input_file=input_file,
                out_path=out_prefix,
                fmt=fmt,
                level=level,
                threshold=threshold,
                labels=labels,
                esm3_offsets=esm3_offsets,
            )
        except Exception as e:
            logger.error(f"Failed {filename}: {e}")


if __name__ == "__main__":
    import config.logging_config

    parser = argparse.ArgumentParser(
        description=(
            "Compute Ca-based residue neighbor lists for every structure file "
            "in a directory.  Outputs one or more TSV files per structure into "
            "the output directory.  Already-processed structures are skipped."
        )
    )
    parser.add_argument(
        "--pdb_path", required=True,
        help="Directory containing input structure files"
    )
    parser.add_argument(
        "--out_path", required=True,
        help="Output directory for neighbor TSV files"
    )
    parser.add_argument(
        "--format", "-f", choices=["pdb", "cif"], default="cif",
        help="Input file format (default: cif)"
    )
    parser.add_argument(
        "--level", choices=["chain", "assembly"], default="chain",
        help=(
            "Calculation level: 'chain' produces one TSV per chain; "
            "'assembly' produces one TSV per structure (default: chain)"
        )
    )
    parser.add_argument(
        "--threshold", type=float, default=8.0,
        help="Ca–Ca distance threshold in Ångströms (default: 8.0)"
    )
    parser.add_argument(
        "--labels", action="store_true",
        help="Annotate indices with chain ID and residue number"
    )
    parser.add_argument(
        "--esm3-offsets", action="store_true",
        help="Remap residue indices to match ESM3 embedding positions"
    )
    args = parser.parse_args()

    process_directory(
        pdb_path=args.pdb_path,
        out_path=args.out_path,
        fmt=args.format,
        level=args.level,
        threshold=args.threshold,
        labels=args.labels,
        esm3_offsets=args.esm3_offsets,
    )
