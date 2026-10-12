#!/usr/bin/env python
"""P0.6 census, step 2: ESM3 + M8_s0 encoders for every census item (research pipeline, unchanged).

The research encode path is IMPORTED, not copied: /pscratch/sd/j/jsegura/embcomp/evalue/encode/encode.py
(md5 d22a8e80b7d84039f67ded9eb5121ee4): SequenceOnlyESM3 bf16 autocast (store settings), make_batches(16000, 128, 40M)
+ assign_batches over the GPUs, numpy float16 cast, sw_common.Whitener (zcw50_clean), f_q -> fp16 q, f_t -> sign ->
np.packbits codes, in 32768-row chunks, TF32 off.  Each item gets BOTH its query encoding q and its target code, so it
can be scored as a query and as a target.

    python census_encode.py ITEMS_FASTA OUT_NPZ
Output npz: ids, lengths, offsets, q fp16 [N_res, 64], codes uint8 [N_res, 8], seq_concat uint8, maxabs, tok_ok, unk,
meta_json.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
import torch.multiprocessing as mp

sys.path.insert(0, "/pscratch/sd/j/jsegura/embcomp/evalue/encode")
os.environ.setdefault("ENCODE_WORK", os.path.dirname(os.path.abspath(__file__)))
import encode as E  # noqa: E402  (research encode.py; imports sw_common and esm3_sequence_embeddings)

W = os.path.dirname(os.path.abspath(__file__))


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def worker(rank, device_name, batches, pdir):
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "4"))))
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    E.fp_setup()
    t0 = time.time()
    embedder = E.S3.SequenceOnlyESM3(device, precision=E.PRECISION)
    enc = E.Enc(device)
    E.log(f"[rank {rank}] models loaded in {time.time() - t0:.1f}s; {len(batches)} batches")
    keys, lens, qs, cs, mx, ok, unk = [], [], [], [], [], [], []
    tes = tenc = 0.0
    for bi, batch in enumerate(batches):
        seqs = [s for _, s in batch]
        t1 = time.time()
        embs = E.embed_batch(embedder, seqs)
        t2 = time.time()
        q, c = enc(np.concatenate(embs))
        tes += t2 - t1
        tenc += time.time() - t2
        o = 0
        for (k, s), e in zip(batch, embs):
            L = len(s)
            a, u = E.token_check(embedder, s)
            keys.append(k)
            lens.append(L)
            qs.append(q[o:o + L])
            cs.append(c[o:o + L])
            mx.append(float(np.abs(e.astype(np.float32)).max()))
            ok.append(a)
            unk.append(u)
            o += L
        assert o == len(q)
        if bi % 50 == 0:
            E.log(f"[rank {rank}] batch {bi + 1}/{len(batches)} ({len(batch)} x {len(batch[0][1])})")
    np.savez(os.path.join(pdir, f"r{rank}.npz"), keys=np.array(keys), lengths=np.array(lens, np.int64),
             q=np.concatenate(qs), codes=np.concatenate(cs), maxabs=np.array(mx, np.float32),
             tok_ok=np.array(ok, np.int8), unk=np.array(unk, np.int32),
             stats=np.array(json.dumps({"rank": rank, "esm_s": round(tes, 1), "enc_s": round(tenc, 1),
                                        "wall_s": round(time.time() - t0, 1),
                                        "peak_gib": round(torch.cuda.max_memory_allocated(device) / 2 ** 30, 2)
                                        if device.type == "cuda" else None})))


def main():
    merge_only = sys.argv[1] == "merge"
    if merge_only:
        sys.argv.pop(1)
    fasta, out = sys.argv[1], sys.argv[2]
    recs = E.read_fasta(fasta)
    ids = [h.split()[0] for h, _ in recs]
    seqs = [s for _, s in recs]
    assert len(set(seqs)) == len(seqs), "items must be unique sequences"
    items = [(i, s) for i, s in zip(ids, seqs)]
    n_dev = torch.cuda.device_count()
    devices = [f"cuda:{i}" for i in range(n_dev)] or ["cpu"]
    batches = E.S3.make_batches(items, E.MAX_TOKENS, E.MAX_BATCH, E.MAX_PAIRS)
    shards = E.S3.assign_batches(batches, len(devices))
    pdir = os.path.join(W, "enc_parts")
    os.makedirs(pdir, exist_ok=True)
    if not merge_only:
        for f in glob.glob(os.path.join(pdir, "*.npz")):
            os.remove(f)
    E.log(f"{len(items)} items, {sum(map(len, seqs)):,} residues, {len(batches)} batches over {devices}")
    t0 = time.time()
    if merge_only:
        devices = ["merge-only run; embedding devices in rank_stats (4 x A100 GPU job)"]
    elif len(devices) == 1:
        worker(0, devices[0], shards[0], pdir)
    else:
        ctx = mp.get_context("spawn")
        ps = [ctx.Process(target=worker, args=(r, d, shards[r], pdir)) for r, d in enumerate(devices)]
        for p in ps:
            p.start()
        for p in ps:
            p.join()
        bad = [r for r, p in enumerate(ps) if p.exitcode != 0]
        if bad:
            raise SystemExit(f"ranks {bad} failed")
    got, stats = {}, []
    for f in sorted(glob.glob(os.path.join(pdir, "r*.npz"))):
        with np.load(f) as z:     # read every array ONCE (z[k] re-reads the member on each access)
            zk = {k: z[k] for k in z.files}
        off = np.concatenate([[0], np.cumsum(zk["lengths"])])
        for k, key in enumerate(zk["keys"].astype(str)):
            got[key] = (zk["q"][off[k]:off[k + 1]], zk["codes"][off[k]:off[k + 1]], float(zk["maxabs"][k]),
                        int(zk["tok_ok"][k]), int(zk["unk"][k]))
        stats.append(json.loads(str(zk["stats"])))
    assert set(got) == set(ids), (len(got), len(ids))
    lengths = np.array([len(s) for s in seqs], np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    q = np.concatenate([got[i][0] for i in ids])
    codes = np.concatenate([got[i][1] for i in ids])
    assert q.shape == (offsets[-1], 64) and codes.shape == (offsets[-1], 8)
    assert all(len(got[i][0]) == L for i, L in zip(ids, lengths))
    maxabs = np.array([got[i][2] for i in ids], np.float32)
    tok_ok = np.array([got[i][3] for i in ids], np.int8)
    unk = np.array([got[i][4] for i in ids], np.int32)
    import esm
    meta = {"fasta": fasta, "fasta_md5": md5(fasta), "n_items": len(ids), "n_residues": int(offsets[-1]),
            "encode_py": E.__file__, "encode_py_md5": md5(E.__file__), "model": E.MODEL, "model_md5": md5(E.MODEL),
            "precision": E.PRECISION, "max_tokens": E.MAX_TOKENS, "max_batch": E.MAX_BATCH, "max_pairs": E.MAX_PAIRS,
            "esm_version": getattr(esm, "__version__", None), "torch": torch.__version__, "devices": devices,
            "n_token_length_mismatch": int((tok_ok == 0).sum()), "n_unk_tokens": int(unk.sum()),
            "max_abs_embedding": float(maxabs.max()), "wall_s": round(time.time() - t0, 1), "rank_stats": stats,
            "slurm_job": os.environ.get("SLURM_JOB_ID")}
    np.savez(out, ids=np.array(ids), lengths=lengths, offsets=offsets, q=q, codes=codes,
             seq_concat=np.frombuffer("".join(seqs).encode("ascii"), np.uint8), maxabs=maxabs, tok_ok=tok_ok, unk=unk,
             meta_json=np.array(json.dumps(meta)))
    E.log("wrote", out, json.dumps({k: v for k, v in meta.items() if k != "rank_stats"}))


if __name__ == "__main__":
    main()
