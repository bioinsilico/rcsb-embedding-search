from __future__ import annotations

import argparse
import logging
import os

logger = logging.getLogger(__name__)


def embedding_exists(domain, embedding_path, ext, cache):
    """Whether <embedding_path>/<domain>.<ext> exists, memoised per domain."""
    if domain not in cache:
        cache[domain] = os.path.isfile(os.path.join(embedding_path, f"{domain}.{ext}"))
    return cache[domain]


def sanitize(tm_score_file, embedding_path, out_file, ext="pt", missing_file=None):
    """Copy rows of a TM-score CSV (domain_i,domain_j,score; no header) to out_file,
    keeping only those whose *both* domains have an embedding file in embedding_path.

    Rows are streamed and written verbatim, so the score text is preserved exactly.
    """
    cache = {}
    missing = set()
    total = kept = malformed = 0

    with open(tm_score_file) as fin, open(out_file, "w") as fout:
        for line in fin:
            row = line.strip()
            if not row:
                continue
            total += 1
            fields = row.split(",")
            if len(fields) < 2:
                malformed += 1
                logger.warning(f"Skipping malformed line {total}: {row!r}")
                continue
            domain_i, domain_j = fields[0], fields[1]
            ok_i = embedding_exists(domain_i, embedding_path, ext, cache)
            ok_j = embedding_exists(domain_j, embedding_path, ext, cache)
            if ok_i and ok_j:
                fout.write(row + "\n")
                kept += 1
            else:
                if not ok_i:
                    missing.add(domain_i)
                if not ok_j:
                    missing.add(domain_j)

    dropped = total - kept - malformed
    logger.info(
        f"Rows: {total} total, {kept} kept, {dropped} dropped (missing embedding), "
        f"{malformed} malformed"
    )
    logger.info(
        f"Unique domains: {len(cache)} checked, {len(missing)} missing embeddings"
    )
    if missing_file is not None:
        with open(missing_file, "w") as f:
            for domain in sorted(missing):
                f.write(domain + "\n")
        logger.info(f"Wrote {len(missing)} missing domain ids -> {missing_file}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    parser = argparse.ArgumentParser(
        description="Keep only TM-score rows whose two domains both have an "
                    "embedding file in the given folder."
    )
    parser.add_argument("--tm_score_file", type=str, required=True,
                        help="Input TM-score CSV (domain_i,domain_j,score; no header).")
    parser.add_argument("--embedding_path", type=str, required=True,
                        help="Folder containing <domain>.<ext> embedding files.")
    parser.add_argument("--out_file", type=str, required=True,
                        help="Output CSV with the surviving rows.")
    parser.add_argument("--ext", type=str, default="pt",
                        help="Embedding file extension (default: pt).")
    parser.add_argument("--missing_file", type=str, default=None,
                        help="Optional: write the sorted unique domain ids whose "
                             "embedding is missing.")
    args = parser.parse_args()

    sanitize(
        args.tm_score_file,
        args.embedding_path,
        args.out_file,
        ext=args.ext,
        missing_file=args.missing_file,
    )
