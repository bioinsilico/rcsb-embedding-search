"""Round-trip / equivalence tests for the packed embedding store.

Runs without a test framework::

    PYTHONPATH=src python src/dataset/tests/test_packed_embeddings.py

Exits non-zero on the first failed assertion. Uses only synthetic data, so it
runs anywhere numpy + torch + pandas are installed (no real embeddings needed).
"""

import os
import sys
import tempfile

import numpy as np
import torch

# Make `src/` importable when run directly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dataset.packed_tm_score_from_embeddings_dataset import PackedTmScoreFromEmbeddingsDataset
from dataset.tm_score_from_embeddings_dataset import TmScoreFromEmbeddingsDataset
from dataset.utils.packed_embeddings import (
    PackedEmbeddingStore,
    build_packed_store,
    is_complete_store,
    stage_store_to_local,
)
from dataset.utils.tm_score_weight import fraction_score, tm_score_weights
from dataset.utils.tools import collate_fn


DIM = 32


def _write_pt_files(pt_dir, rng, dtype=torch.float32):
    """Create synthetic per-domain embeddings plus deliberately-broken files.

    Returns a dict of the *good* domains -> their reference tensors.
    """
    good = {}
    lengths = {"a": 1, "b": 5, "c": 17, "d": 128, "e": 3}
    for dom, length in lengths.items():
        t = torch.from_numpy(rng.standard_normal((length, DIM)).astype(np.float32)).to(dtype)
        torch.save(t, os.path.join(pt_dir, f"{dom}.pt"))
        good[dom] = t

    # empty embedding (0 residues) -> must be skipped
    torch.save(torch.zeros((0, DIM), dtype=dtype), os.path.join(pt_dir, "empty.pt"))
    # corrupt file -> torch.load raises -> must be skipped
    with open(os.path.join(pt_dir, "corrupt.pt"), "wb") as fh:
        fh.write(b"not a real torch file")
    # "gone" is referenced by a pair but has no file at all -> must be skipped
    return good


def _write_pairs(path, pairs):
    with open(path, "w") as fh:
        for i, j, s in pairs:
            fh.write(f"{i},{j},{s}\n")


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def test_build_and_roundtrip(tmp):
    rng = np.random.default_rng(0)
    pt_dir = os.path.join(tmp, "pt")
    os.makedirs(pt_dir)
    good = _write_pt_files(pt_dir, rng)

    store_dir = os.path.join(tmp, "store")
    domain_to_path = {
        d: os.path.join(pt_dir, f"{d}.pt")
        for d in ["a", "b", "c", "d", "e", "empty", "corrupt", "gone"]
    }
    summary = build_packed_store(domain_to_path, store_dir)

    check(is_complete_store(store_dir), "store dir incomplete after build")
    check(summary["count"] == len(good), f"expected {len(good)} packed, got {summary['count']}")
    check(set(summary["missing"]) == {"empty", "corrupt", "gone"},
          f"unexpected missing set: {summary['missing']}")
    check(summary["dim"] == DIM, f"dim mismatch: {summary['dim']}")

    store = PackedEmbeddingStore(store_dir)
    for dom, ref in good.items():
        got = store.get(dom)
        check(got.shape == ref.shape, f"{dom}: shape {got.shape} != {ref.shape}")
        check(got.dtype == torch.float32, f"{dom}: dtype {got.dtype}")
        check(torch.allclose(got, ref), f"{dom}: values differ after round-trip")
    check("empty" not in store and "gone" not in store, "skipped domain present in store")
    # returned tensor must be writable (detached from the read-only mmap)
    w = store.get("b")
    w[0, 0] = 123.0
    check(store.get("b")[0, 0] != 123.0, "mutation leaked back into the store")
    print("[ok] build + round-trip + skip-missing + writability")
    return store_dir, pt_dir, good


def test_dataset_equivalence(tmp, store_dir, pt_dir, good):
    # A pair file over good domains only, so packed and original keep identical rows.
    good_pairs = [
        ("a", "b", 0.9), ("b", "c", 0.4), ("c", "d", 0.75),
        ("d", "e", 0.2), ("e", "a", 0.6), ("a", "c", 0.85),
    ]
    pairs_path = os.path.join(tmp, "good_pairs.csv")
    _write_pairs(pairs_path, good_pairs)

    packed = PackedTmScoreFromEmbeddingsDataset(
        pairs_path, store_dir,
        score_method=fraction_score,
        weighting_method=tm_score_weights(5, 0.25),
    )
    original = TmScoreFromEmbeddingsDataset(
        pairs_path, pt_dir,
        score_method=fraction_score,
        weighting_method=tm_score_weights(5, 0.25),
    )
    check(len(packed) == len(original) == len(good_pairs),
          f"length mismatch packed={len(packed)} original={len(original)}")
    check(np.allclose(packed.weights(), original.weights()), "weights differ")

    for idx in range(len(packed)):
        (px, py, pz) = packed[idx]
        (ox, oy, oz) = original[idx]
        check(torch.allclose(px, ox), f"item {idx}: x embeddings differ")
        check(torch.allclose(py, oy), f"item {idx}: y embeddings differ")
        check(torch.allclose(pz, oz), f"item {idx}: label differs")
    print("[ok] packed dataset == original dataset (item-for-item)")
    return pairs_path


def test_drop_missing(tmp, store_dir):
    # Pairs that reference skipped domains must be dropped, not crash.
    pairs = [("a", "b", 0.9), ("a", "gone", 0.5), ("empty", "c", 0.3), ("c", "d", 0.7)]
    pairs_path = os.path.join(tmp, "mixed_pairs.csv")
    _write_pairs(pairs_path, pairs)
    ds = PackedTmScoreFromEmbeddingsDataset(
        pairs_path, store_dir, score_method=fraction_score,
        weighting_method=tm_score_weights(5, 0.25),
    )
    check(len(ds) == 2, f"expected 2 surviving pairs, got {len(ds)}")
    for idx in range(len(ds)):
        x, y, z = ds[idx]  # must not raise
        check(x.ndim == 2 and y.ndim == 2, "bad item shape")
    print("[ok] pairs referencing missing domains are dropped")


def test_dataloader_workers(tmp, store_dir, pairs_path):
    from torch.utils.data import DataLoader

    ds = PackedTmScoreFromEmbeddingsDataset(
        pairs_path, store_dir, score_method=fraction_score,
        weighting_method=tm_score_weights(5, 0.25),
    )
    # Force 'fork' (Linux/NERSC default) so we exercise the memmap fork-safety
    # path; macOS defaults to 'spawn', which would pickle the dataset's
    # (unpicklable) weighting closure -- unrelated to the store.
    loader = DataLoader(
        ds, batch_size=3, num_workers=2, collate_fn=collate_fn,
        multiprocessing_context="fork",
    )
    seen = 0
    for (x, x_mask), (y, y_mask), z in loader:
        check(x.shape[0] == z.shape[0], "batch/label size mismatch")
        check(x.shape[2] == DIM and y.shape[2] == DIM, "embedding dim mismatch in batch")
        seen += z.shape[0]
    check(seen == len(ds), f"dataloader yielded {seen} items, expected {len(ds)}")
    print("[ok] DataLoader with num_workers=2 (memmap is fork-safe)")


def test_staging(tmp, store_dir, good):
    local_dir = os.path.join(tmp, "local")
    os.makedirs(local_dir)
    staged = stage_store_to_local(store_dir, local_dir)
    check(staged != store_dir, "staging did not relocate the store")
    check(is_complete_store(staged), "staged store incomplete")
    # second call is idempotent and returns the same staged dir
    check(stage_store_to_local(store_dir, local_dir) == staged, "staging not idempotent")
    store = PackedEmbeddingStore(staged)
    for dom, ref in good.items():
        check(torch.allclose(store.get(dom), ref), f"{dom}: staged copy differs")
    print("[ok] local staging (per-node copy + idempotent) round-trips")


def test_bf16_upcast(tmp):
    rng = np.random.default_rng(1)
    pt_dir = os.path.join(tmp, "pt_bf16")
    os.makedirs(pt_dir)
    ref = torch.from_numpy(rng.standard_normal((7, DIM)).astype(np.float32)).to(torch.bfloat16)
    torch.save(ref, os.path.join(pt_dir, "x.pt"))
    store_dir = os.path.join(tmp, "store_bf16")
    build_packed_store({"x": os.path.join(pt_dir, "x.pt")}, store_dir)
    store = PackedEmbeddingStore(store_dir)
    got = store.get("x")
    check(got.dtype == torch.float32, f"bf16 not upcast to float32 (got {got.dtype})")
    check(torch.allclose(got, ref.float()), "bf16 upcast values differ")
    print("[ok] bfloat16 embeddings upcast to float32 losslessly")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        store_dir, pt_dir, good = test_build_and_roundtrip(tmp)
        pairs_path = test_dataset_equivalence(tmp, store_dir, pt_dir, good)
        test_drop_missing(tmp, store_dir)
        test_dataloader_workers(tmp, store_dir, pairs_path)
        test_staging(tmp, store_dir, good)
        test_bf16_upcast(tmp)
    print("\nALL PACKED-EMBEDDING TESTS PASSED")


if __name__ == "__main__":
    main()
