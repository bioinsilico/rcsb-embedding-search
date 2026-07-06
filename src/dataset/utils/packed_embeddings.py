"""Packed, memory-mapped embedding store.

Motivation
----------
The original :class:`~dataset.tm_score_from_embeddings_dataset.TmScoreFromEmbeddingsDataset`
reads two individual ``{domain}.pt`` files from the shared filesystem on *every*
``__getitem__``. At scale (hundreds of ranks, each with several dataloader
workers) this turns into a storm of random small-file ``open()`` calls against a
parallel filesystem's metadata servers. A single slow/hung read makes one rank
miss its gradient all-reduce, which stalls the whole job until the NCCL watchdog
kills it (30 min default timeout).

This module packs every per-residue embedding into a *single* flat binary blob
plus a compact numpy index. At training time the blob is memory-mapped once and
each sample is served by slicing a contiguous span of rows -- no per-sample file
opens. The blob can optionally be staged to node-local storage first (see
:func:`stage_store_to_local`) so reads never touch the shared filesystem during
training.

On-disk layout of a store directory
-----------------------------------
* ``embeddings.dat`` -- raw row-major ``(total_rows, dim)`` array of ``meta.dtype``.
* ``index.npz``      -- ``domains`` (unicode), ``offsets`` (int64 rows),
  ``lengths`` (int64 rows).
* ``meta.json``      -- ``dim``, ``dtype``, ``total_rows``, ``count``,
  ``format_version``.

The format is numpy-native on purpose so it has no extra dependencies (no
parquet/pyarrow) and loads identically on the login node and the compute nodes.
"""

import errno
import fcntl
import hashlib
import json
import logging
import os
import shutil

import numpy as np
import torch

logger = logging.getLogger(__name__)

DATA_NAME = "embeddings.dat"
INDEX_NAME = "index.npz"
META_NAME = "meta.json"
STORE_FILES = (DATA_NAME, INDEX_NAME, META_NAME)
FORMAT_VERSION = 1

# numpy has no bfloat16, so bf16 embeddings are upcast to float32 on packing.
_TORCH_TO_NUMPY = {
    torch.float32: np.float32,
    torch.float16: np.float16,
    torch.bfloat16: np.float32,
    torch.float64: np.float32,
}
_NUMPY_TO_TORCH = {
    np.dtype(np.float32): torch.float32,
    np.dtype(np.float16): torch.float16,
}


def _numpy_dtype_for(torch_dtype):
    if torch_dtype not in _TORCH_TO_NUMPY:
        raise TypeError(f"Unsupported embedding dtype {torch_dtype}; expected a float tensor")
    return np.dtype(_TORCH_TO_NUMPY[torch_dtype])


def _to_numpy(tensor, np_dtype):
    """Return a C-contiguous numpy view of ``tensor`` cast to ``np_dtype``.

    Cast happens in torch first (so bfloat16 -> float32 works, which numpy can't
    do directly), then a contiguous numpy array is produced.
    """
    torch_dtype = _NUMPY_TO_TORCH[np.dtype(np_dtype)]
    arr = tensor.detach().to(dtype=torch_dtype, device="cpu").contiguous().numpy()
    return np.ascontiguousarray(arr)


def build_packed_store(domain_to_path, out_dir, target_dtype=None, overwrite=False, log_every=50000):
    """Pack ``.pt`` embeddings referenced by ``domain_to_path`` into ``out_dir``.

    Each source file is read exactly once (single pass). Domains whose file is
    missing, unreadable, not a 2-D tensor, or empty (``L == 0``) are skipped and
    returned in ``summary['missing']`` so callers can drop pairs that reference
    them.

    Args:
        domain_to_path: mapping ``{domain_id: /path/to/domain.pt}``. Iteration
            order is preserved and becomes the on-disk order.
        out_dir: destination directory for the store (created if needed).
        target_dtype: numpy dtype for the blob. If ``None`` it is inferred from
            the first successfully-loaded tensor (bfloat16 -> float32).
        overwrite: rebuild even if a complete store already exists in ``out_dir``.
        log_every: progress log cadence (in domains).

    Returns:
        dict summary with keys ``count``, ``missing``, ``total_rows``, ``dim``,
        ``dtype``, ``out_dir``.
    """
    os.makedirs(out_dir, exist_ok=True)
    if not overwrite and is_complete_store(out_dir):
        meta = _read_meta(out_dir)
        logger.info(f"Packed store already present at {out_dir} ({meta['count']} domains); skipping build")
        return {
            "count": meta["count"],
            "missing": [],
            "total_rows": meta["total_rows"],
            "dim": meta["dim"],
            "dtype": meta["dtype"],
            "out_dir": out_dir,
        }

    data_path = os.path.join(out_dir, DATA_NAME)
    data_tmp = data_path + ".tmp"

    np_dtype = np.dtype(target_dtype) if target_dtype is not None else None
    dim = None
    domains, offsets, lengths, missing = [], [], [], []
    cur_rows = 0

    total = len(domain_to_path)
    with open(data_tmp, "wb", buffering=1024 * 1024) as fh:
        for i, (domain, path) in enumerate(domain_to_path.items()):
            if i % log_every == 0:
                logger.info(f"Packing {i}/{total} (kept={len(domains)}, missing={len(missing)}, rows={cur_rows})")
            try:
                tensor = torch.load(path, map_location="cpu")
            except Exception as exc:  # missing / corrupt / truncated file
                logger.warning(f"Skipping {domain}: {type(exc).__name__}: {exc}")
                missing.append(domain)
                continue
            if not torch.is_tensor(tensor) or tensor.ndim != 2:
                shape = tuple(tensor.shape) if torch.is_tensor(tensor) else type(tensor).__name__
                logger.warning(f"Skipping {domain}: expected a 2-D tensor, got {shape}")
                missing.append(domain)
                continue
            if tensor.size(0) == 0:
                logger.warning(f"Skipping {domain}: empty embedding (0 residues)")
                missing.append(domain)
                continue

            if np_dtype is None:
                np_dtype = _numpy_dtype_for(tensor.dtype)
            if dim is None:
                dim = int(tensor.size(1))
            elif tensor.size(1) != dim:
                raise ValueError(
                    f"Inconsistent embedding dim for {domain}: got {tensor.size(1)}, expected {dim}"
                )

            arr = _to_numpy(tensor, np_dtype)
            arr.tofile(fh)
            domains.append(domain)
            offsets.append(cur_rows)
            lengths.append(int(arr.shape[0]))
            cur_rows += int(arr.shape[0])

    if not domains:
        # nothing written; clean up and fail loudly rather than leaving a broken store
        if os.path.exists(data_tmp):
            os.remove(data_tmp)
        raise RuntimeError(f"No usable embeddings packed from {total} candidates in {out_dir}")

    os.replace(data_tmp, data_path)

    index_tmp = os.path.join(out_dir, INDEX_NAME + ".tmp")
    with open(index_tmp, "wb") as fh:
        np.savez(
            fh,
            domains=np.array(domains),
            offsets=np.array(offsets, dtype=np.int64),
            lengths=np.array(lengths, dtype=np.int64),
        )
    os.replace(index_tmp, os.path.join(out_dir, INDEX_NAME))

    meta = {
        "format_version": FORMAT_VERSION,
        "dim": int(dim),
        "dtype": np.dtype(np_dtype).name,
        "total_rows": int(cur_rows),
        "count": len(domains),
    }
    _write_meta(out_dir, meta)

    logger.info(
        f"Packed {len(domains)} domains ({cur_rows} rows, dim={dim}, dtype={meta['dtype']}) "
        f"into {out_dir}; {len(missing)} skipped"
    )
    return {
        "count": len(domains),
        "missing": missing,
        "total_rows": cur_rows,
        "dim": dim,
        "dtype": meta["dtype"],
        "out_dir": out_dir,
    }


def _write_meta(out_dir, meta):
    meta_tmp = os.path.join(out_dir, META_NAME + ".tmp")
    with open(meta_tmp, "w") as fh:
        json.dump(meta, fh, indent=2)
    os.replace(meta_tmp, os.path.join(out_dir, META_NAME))


def _read_meta(store_dir):
    with open(os.path.join(store_dir, META_NAME)) as fh:
        return json.load(fh)


def is_complete_store(store_dir):
    """True if ``store_dir`` contains all three store files."""
    return all(os.path.exists(os.path.join(store_dir, name)) for name in STORE_FILES)


class PackedEmbeddingStore:
    """Read-only accessor over a packed embedding store.

    The memory map is opened lazily and re-opened after a fork, so a single
    instance can be constructed in the main process and safely used from
    dataloader workers.
    """

    def __init__(self, store_dir):
        self.store_dir = store_dir
        if not is_complete_store(store_dir):
            missing = [n for n in STORE_FILES if not os.path.exists(os.path.join(store_dir, n))]
            raise FileNotFoundError(f"Incomplete packed store at {store_dir}; missing {missing}")

        meta = _read_meta(store_dir)
        if meta.get("format_version") != FORMAT_VERSION:
            raise ValueError(
                f"Unsupported packed-store format_version {meta.get('format_version')} at {store_dir}"
            )
        self.dim = int(meta["dim"])
        self.dtype = np.dtype(meta["dtype"])
        self.total_rows = int(meta["total_rows"])
        self.data_path = os.path.join(store_dir, DATA_NAME)

        index = np.load(os.path.join(store_dir, INDEX_NAME))
        self._domains = index["domains"].astype(str)
        offsets = index["offsets"]
        lengths = index["lengths"]
        self._offset = {d: int(o) for d, o in zip(self._domains, offsets)}
        self._length = {d: int(length) for d, length in zip(self._domains, lengths)}

        self._mm = None
        self._mm_pid = None

    @property
    def domains(self):
        return self._domains

    def __contains__(self, domain):
        return domain in self._offset

    def __len__(self):
        return len(self._offset)

    def _ensure_mm(self):
        # Re-open after fork: a memmap fd shared across fork is unsafe.
        if self._mm is None or self._mm_pid != os.getpid():
            self._mm = np.memmap(
                self.data_path, dtype=self.dtype, mode="r", shape=(self.total_rows, self.dim)
            )
            self._mm_pid = os.getpid()
        return self._mm

    def get(self, domain):
        """Return the ``(L, dim)`` embedding for ``domain`` as a writable tensor."""
        offset = self._offset[domain]
        length = self._length[domain]
        mm = self._ensure_mm()
        # .copy() detaches from the read-only mmap -> a clean, writable tensor.
        span = np.array(mm[offset:offset + length])
        return torch.from_numpy(span)

    def close(self):
        self._mm = None
        self._mm_pid = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_mm"] = None
        state["_mm_pid"] = None
        return state


def resolve_local_dir(spec):
    """Resolve a staging directory spec to a concrete path (or ``None``).

    ``"auto"`` picks the first of ``$SLURM_TMPDIR``, ``$TMPDIR``, ``/tmp`` that
    exists. Any other non-empty string is returned as-is. ``None``/``""``
    disables staging.
    """
    if not spec:
        return None
    if spec == "auto":
        for env in ("SLURM_TMPDIR", "TMPDIR"):
            val = os.environ.get(env)
            if val and os.path.isdir(val):
                return val
        return "/tmp" if os.path.isdir("/tmp") else None
    return str(spec)


def stage_store_to_local(store_dir, local_dir, min_free_bytes_margin=1 << 30):
    """Copy a packed store to node-local storage, once per node.

    Coordination is done with an exclusive ``flock`` on a lockfile in the
    destination directory, so multiple ranks sharing a node cooperate: the first
    to acquire the lock copies the files, the rest wait and then reuse the copy.
    A ``.complete`` marker makes the operation idempotent across job steps.

    If ``local_dir`` is falsy, the destination is unwritable, or there isn't
    enough free space, the original ``store_dir`` is returned unchanged (the
    store is then memory-mapped from the shared filesystem).

    Returns the directory that should be handed to :class:`PackedEmbeddingStore`.
    """
    local_dir = resolve_local_dir(local_dir)
    if not local_dir:
        return store_dir

    src_abs = os.path.abspath(store_dir)
    tag = hashlib.md5(src_abs.encode()).hexdigest()[:8]
    dest = os.path.join(local_dir, f"packed_emb_{os.path.basename(src_abs.rstrip('/'))}_{tag}")
    marker = dest + ".complete"

    if os.path.exists(marker) and is_complete_store(dest):
        logger.info(f"Using already-staged packed store at {dest}")
        return dest

    try:
        os.makedirs(dest, exist_ok=True)
    except OSError as exc:
        logger.warning(f"Cannot create local staging dir {dest} ({exc}); using shared store at {store_dir}")
        return store_dir

    total_bytes = sum(os.path.getsize(os.path.join(store_dir, n)) for n in STORE_FILES)
    lock_path = dest + ".lock"
    try:
        lock_fh = open(lock_path, "w")
    except OSError as exc:
        logger.warning(f"Cannot open staging lock {lock_path} ({exc}); using shared store at {store_dir}")
        return store_dir

    try:
        fcntl.flock(lock_fh, fcntl.LOCK_EX)
        # Re-check under the lock: another rank on this node may have finished.
        if os.path.exists(marker) and is_complete_store(dest):
            logger.info(f"Using packed store staged by a peer rank at {dest}")
            return dest

        free = shutil.disk_usage(local_dir).free
        if free < total_bytes + min_free_bytes_margin:
            logger.warning(
                f"Not enough space in {local_dir} to stage store "
                f"({free} free < {total_bytes} needed); using shared store at {store_dir}"
            )
            return store_dir

        logger.info(f"Staging packed store {store_dir} -> {dest} ({total_bytes} bytes)")
        for name in STORE_FILES:
            src = os.path.join(store_dir, name)
            tmp = os.path.join(dest, name + ".tmp")
            shutil.copyfile(src, tmp)
            os.replace(tmp, os.path.join(dest, name))
        open(marker, "w").close()
        logger.info(f"Staged packed store ready at {dest}")
        return dest
    except OSError as exc:
        logger.warning(f"Staging to {dest} failed ({exc}); using shared store at {store_dir}")
        return store_dir
    finally:
        try:
            fcntl.flock(lock_fh, fcntl.LOCK_UN)
        except OSError as exc:
            if exc.errno != errno.EBADF:
                raise
        lock_fh.close()
