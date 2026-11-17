"""Build the inputs of ``training/embedding_tm_score_v2.py`` from per-domain ESM3 files and pair CSVs.

Writes to ``--out-dir``:
  store/           packed store (``dataset.utils.packed_embeddings``) of every domain in the pair files;
  train_pairs.npz  ``domains``, ``i``, ``j`` (int32), ``score`` (float32) of the training CSV;
  val_pairs.npz    the same for a random subset of the validation CSV (``--val-pairs`` rows);
  input_stats.npz  per-dimension ``mean`` / ``std`` over the residue rows (BOS/EOS included, as the
                   model sees them) of the training domains, for ``TransformerEmbeddingCosine(input_norm=True)``.

With ``--stats-only --exclude-domains-file F --stats-out P`` it only recomputes the statistics over the
training domains not listed in F (a held-out split must not contribute to its own normalisation).

    python src/scripts/build_tm_pair_store.py --embeddings-dir .../curated-esm3 \\
        --train-csv .../cath_23Mids-TMfast.sanitized.csv --val-csv .../scop_low_sen_test_set.sanitized.csv \\
        --out-dir .../stage1 --dtype float16
"""
import argparse
import logging
import os

import numpy as np
import pandas as pd

from dataset.utils.packed_embeddings import PackedEmbeddingStore, build_packed_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def read_pairs(path):
    return pd.read_csv(path, header=None, names=["a", "b", "s"], dtype={"a": str, "b": str, "s": np.float32},
                       keep_default_na=False)


def to_npz(df, domains, out):
    cat = pd.CategoricalDtype(categories=domains)
    i = df["a"].astype(cat).cat.codes.to_numpy()
    j = df["b"].astype(cat).cat.codes.to_numpy()
    ok = (i >= 0) & (j >= 0)
    np.savez(out, domains=np.array(domains), i=i[ok].astype(np.int32), j=j[ok].astype(np.int32),
             score=df["s"].to_numpy(np.float32)[ok])
    logger.info(f"{out}: {int(ok.sum()):,} pairs ({int((~ok).sum()):,} dropped: domain not in the store)")


def input_stats(store, out, domains):
    """Per-dimension mean / std over the stored rows of ``domains`` (float64 accumulation)."""
    mm = store._ensure_mm()
    s = np.zeros(store.dim, dtype=np.float64)
    ss = np.zeros(store.dim, dtype=np.float64)
    max_abs = np.zeros(store.dim, dtype=np.float64)
    n = 0
    for d in domains:
        o, L = store._offset[d], store.length(d)
        x = np.asarray(mm[o:o + L], dtype=np.float64)
        if not np.isfinite(x).all():                   # e.g. float16 overflow (|x| > 65504) at packing
            raise ValueError(f"non-finite values in the stored embedding of {d}")
        s += x.sum(0)
        ss += (x * x).sum(0)
        np.maximum(max_abs, np.abs(x).max(0), out=max_abs)
        n += L
    mean = s / n
    std = np.sqrt(np.maximum(ss / n - mean ** 2, 0.0))
    np.savez(out, mean=mean.astype(np.float32), std=std.astype(np.float32), max_abs=max_abs.astype(np.float32),
             rows=n, domains=len(domains))
    top = np.argsort(-np.abs(mean))[:3]
    logger.info(f"input stats over {n:,} rows of {len(domains):,} domains; largest |mean| dims {top.tolist()} "
                f"mean {mean[top].round(1).tolist()} std {std[top].round(1).tolist()}; max |x| {max_abs.max():.1f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embeddings-dir", required=True)
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--val-csv", required=True)
    ap.add_argument("--val-pairs", type=int, default=200_000)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stats-only", action="store_true")
    ap.add_argument("--exclude-domains-file", default=None)
    ap.add_argument("--stats-out", default=None)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    if a.stats_only:
        store = PackedEmbeddingStore(os.path.join(a.out_dir, "store"))
        z = np.load(os.path.join(a.out_dir, "train_pairs.npz"))
        names = z["domains"]
        used = np.unique(np.concatenate([z["i"], z["j"]]))
        excluded = set()
        if a.exclude_domains_file:
            excluded = {l.strip() for l in open(a.exclude_domains_file) if l.strip()}
        domains = [str(names[k]) for k in used if str(names[k]) not in excluded]
        logger.info(f"statistics over {len(domains):,} training domains ({len(excluded)} excluded)")
        input_stats(store, a.stats_out, domains)
        return

    train = read_pairs(a.train_csv)
    val = read_pairs(a.val_csv)
    val = val.sample(n=min(a.val_pairs, len(val)), random_state=a.seed)
    names = pd.unique(pd.concat([train["a"], train["b"], val["a"], val["b"]]))
    paths = {d: os.path.join(a.embeddings_dir, f"{d}.pt") for d in names}
    logger.info(f"{len(train):,} training pairs, {len(val):,} validation pairs, {len(paths):,} domains")

    store_dir = os.path.join(a.out_dir, "store")
    summary = build_packed_store(paths, store_dir, target_dtype=np.dtype(a.dtype))
    logger.info(f"store: {summary['count']} domains, {summary['total_rows']:,} rows, {len(summary['missing'])} missing")
    store = PackedEmbeddingStore(store_dir)
    domains = [str(d) for d in store.domains]
    to_npz(train, domains, os.path.join(a.out_dir, "train_pairs.npz"))
    to_npz(val, domains, os.path.join(a.out_dir, "val_pairs.npz"))
    used = np.unique(np.concatenate([np.load(os.path.join(a.out_dir, "train_pairs.npz"))[k] for k in ("i", "j")]))
    input_stats(store, os.path.join(a.out_dir, "input_stats.npz"), [domains[k] for k in used])


if __name__ == "__main__":
    main()
