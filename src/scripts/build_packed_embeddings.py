"""Build a packed, memory-mapped embedding store from individual ``.pt`` files.

Reads the domain ids referenced by one or more TM-score pair files, resolves
each to ``{embedding_path}/{domain}.pt``, and packs them into a single store
directory (blob + numpy index + meta) consumable by
:class:`~dataset.packed_tm_score_from_embeddings_dataset.PackedTmScoreFromEmbeddingsDataset`.

Run this once (e.g. on a login node or in a short prep job) before training::

    python src/scripts/build_packed_embeddings.py \
        --embedding_path /path/to/pt_dir \
        --tm_score_file train_pairs.csv validation_pairs.csv \
        --out_dir /path/to/packed_store

Missing/corrupt/empty embeddings are skipped and written to
``{out_dir}/missing_domains.txt``.
"""

import argparse
import logging
import os

import pandas as pd

from dataset.utils.packed_embeddings import build_packed_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def collect_domains(tm_score_files):
    """Return the unique domain ids referenced across the given pair files, in first-seen order."""
    seen = {}
    for path in tm_score_files:
        df = pd.read_csv(
            path,
            header=None,
            index_col=None,
            names=['domain_i', 'domain_j', 'score'],
            dtype={'domain_i': 'str', 'domain_j': 'str', 'score': 'float32'},
        )
        for domain in pd.concat([df['domain_i'], df['domain_j']]).tolist():
            seen.setdefault(domain, None)
    return list(seen.keys())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--embedding_path', type=str, required=True,
                        help='Directory containing per-domain {domain}.pt embedding files')
    parser.add_argument('--tm_score_file', type=str, nargs='+', required=True,
                        help='One or more TM-score pair CSVs (domain_i,domain_j,score)')
    parser.add_argument('--out_dir', type=str, required=True, help='Destination store directory')
    parser.add_argument('--dtype', type=str, default=None, choices=[None, 'float32', 'float16'],
                        help='Blob dtype (default: infer from data; bfloat16 -> float32)')
    parser.add_argument('--overwrite', action='store_true', help='Rebuild even if a complete store exists')
    args = parser.parse_args()

    domains = collect_domains(args.tm_score_file)
    logger.info(f"Collected {len(domains)} unique domains from {len(args.tm_score_file)} pair file(s)")

    domain_to_path = {d: os.path.join(args.embedding_path, f"{d}.pt") for d in domains}
    summary = build_packed_store(
        domain_to_path,
        args.out_dir,
        target_dtype=args.dtype,
        overwrite=args.overwrite,
    )

    if summary['missing']:
        missing_path = os.path.join(args.out_dir, 'missing_domains.txt')
        with open(missing_path, 'w') as fh:
            fh.write('\n'.join(summary['missing']) + '\n')
        logger.warning(f"{len(summary['missing'])} domains skipped; ids written to {missing_path}")

    logger.info(
        f"Done: {summary['count']} domains, {summary['total_rows']} rows, "
        f"dim={summary['dim']}, dtype={summary['dtype']} -> {summary['out_dir']}"
    )


if __name__ == '__main__':
    main()
