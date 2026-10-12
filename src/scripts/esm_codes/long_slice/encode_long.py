#!/usr/bin/env python
"""P0.7 long slice, step 2 (NERSC GPU): M8_s0 (= L2m_B8_s0) query encodings AND target codes of the sampled proteins
(long_q.fasta, long_t.fasta, long_r.fasta from select_long.py), through the research pipeline that produced every
encoding the shipped calibration was fitted on.

evalue/encode/encode.py (md5 d22a8e80b7d84039f67ded9eb5121ee4) is IMPORTED, not copied, exactly as panel/score2's
encode_panel.py did: ESM3-open sequence-only batched forward (src/scripts/esm3_sequence_embeddings.py SequenceOnlyESM3,
bf16 autocast, store batching make_batches(16000 tokens, 128, 40M pairs), last block before the final LN, BOS/EOS
removed) -> fp16 -> zcw50_clean -> f_q (fp16 q) and f_t -> sign -> packbits (codes), encoders on the concatenated batch in
32768-row chunks (E.Enc). The only difference from encode.py's own worker: BOTH q and codes are stored for every
protein, so any set can be used as queries or as targets.

    python encode_long.py embed long_q long_t long_r   # all visible GPUs, resumable (enc/parts/<set>/), then merge
    python encode_long.py merge long_q long_t long_r   #  -> enc/out/<set>.L2m_B8_s0.npz (q fp16, codes uint8, offsets..)
    python encode_long.py verify                       # GPU 0, after merge, separate process:
        (a) 20 benchmark targets (default_rng(20260927), as panel/score2 verify (a)) re-encoded -> vs the stored
            benchmark codes (evalue/encode/out/targets.L2m_B8_s0.npz): the pipeline is the benchmark's;
        (b) 12 long proteins (3 per long bin of long_q, default_rng(7)) re-encoded in a new process (other batch
            composition) -> code bits vs production; and the UNBATCHED REFERENCE PATH (SequenceOnlyESM3.reference =
            ESM SDK forward_and_sample, geometric attention on, what FoldMatch runs): residue cosine vs the batched
            re-embedding, codes vs production codes, peak GPU memory; OOM is recorded, not fatal.
        -> enc/results/verify_long.<job>.json
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("ENCODE_WORK", os.path.join(HERE, "enc"))
ENCODE_DIR = "/pscratch/sd/j/jsegura/embcomp/evalue/encode"
if ENCODE_DIR not in sys.path:
    sys.path.insert(0, ENCODE_DIR)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402

import encode as E  # noqa: E402

assert os.path.abspath(E.W) == os.path.abspath(os.environ["ENCODE_WORK"]), (E.W, os.environ["ENCODE_WORK"])
ENCODE_MD5 = "d22a8e80b7d84039f67ded9eb5121ee4"
SETS = {k: os.path.join(HERE, f"{k}.fasta") for k in ("long_q", "long_t", "long_r")}
BENCH_T = os.path.join(ENCODE_DIR, "out", "targets.L2m_B8_s0.npz")
FLUSH = 20


def out_path(set_name):
    return os.path.join(E.W, "out", f"{set_name}.{E.RUN}.npz")


def unique_items(set_name):
    recs = E.read_fasta(SETS[set_name])
    uniq = {}
    for _, s in recs:
        uniq.setdefault(E.key_of(s), s)
    return recs, uniq


def worker(rank, device_name, jobs, tag, deadline):
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "4"))))
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    E.fp_setup()
    t0 = time.time()
    embedder = E.S3.SequenceOnlyESM3(device, precision=E.PRECISION)
    enc = E.Enc(device)
    E.log(f"[rank {rank}] models loaded in {time.time() - t0:.1f} s on {device_name}")
    stats = {"rank": rank, "load_s": round(time.time() - t0, 1), "sets": {}}
    for set_name, batches in jobs:
        pdir = os.path.join(E.W, "parts", set_name)
        os.makedirs(pdir, exist_ok=True)
        buf = {k: [] for k in ("keys", "lengths", "q", "codes", "maxabs", "tok_ok", "unk")}
        st = {"batches": 0, "seqs": 0, "res": 0, "esm_s": 0.0, "enc_s": 0.0, "stopped": False, "max_len": 0}
        chunk = 0
        ts = time.time()

        def flush():
            nonlocal chunk
            if not buf["keys"]:
                return
            blob = {"keys": np.array(buf["keys"]), "lengths": np.array(buf["lengths"], np.int64),
                    "maxabs": np.array(buf["maxabs"], np.float32), "tok_ok": np.array(buf["tok_ok"], np.int8),
                    "unk": np.array(buf["unk"], np.int32), "q": np.concatenate(buf["q"]),
                    "codes": np.concatenate(buf["codes"])}
            fn = os.path.join(pdir, f"{tag}_r{rank}_c{chunk:04d}.npz")
            with open(fn + ".tmp", "wb") as fh:
                np.savez(fh, **blob)
            os.replace(fn + ".tmp", fn)
            chunk += 1
            for v in buf.values():
                v.clear()

        for bi, batch in enumerate(batches):
            if deadline and time.time() > deadline:
                st["stopped"] = True
                E.log(f"[rank {rank}] {set_name}: deadline reached after {bi} / {len(batches)} batches")
                break
            seqs = [s for _, s in batch]
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.time()
            embs = E.embed_batch(embedder, seqs)
            t2 = time.time()
            q, codes = enc(np.concatenate(embs))
            if device.type == "cuda":
                torch.cuda.synchronize()
            t3 = time.time()
            st["esm_s"] += t2 - t1
            st["enc_s"] += t3 - t2
            o = 0
            for (k, s), e in zip(batch, embs):
                L = len(s)
                ok, unk = E.token_check(embedder, s)
                buf["keys"].append(k)
                buf["lengths"].append(L)
                buf["maxabs"].append(float(np.abs(e.astype(np.float32)).max()))
                buf["tok_ok"].append(ok)
                buf["unk"].append(unk)
                buf["q"].append(q[o:o + L])
                buf["codes"].append(codes[o:o + L])
                st["max_len"] = max(st["max_len"], L)
                o += L
            assert o == len(q) == len(codes)
            st["batches"] += 1
            st["seqs"] += len(batch)
            st["res"] += o
            if (bi + 1) % FLUSH == 0:
                flush()
            if bi % 25 == 0 or bi == len(batches) - 1:
                el = time.time() - ts
                E.log(f"[rank {rank}] {set_name} batch {bi + 1}/{len(batches)} ({len(batch)} x {len(batch[0][1])}): "
                      f"{st['seqs']} seqs, {st['res'] / max(el, 1e-9):,.0f} res/s, esm {st['esm_s']:.1f} s, "
                      f"enc {st['enc_s']:.1f} s")
        flush()
        st["wall_s"] = round(time.time() - ts, 1)
        st["esm_s"], st["enc_s"] = round(st["esm_s"], 1), round(st["enc_s"], 1)
        if device.type == "cuda":
            st["peak_alloc_gib"] = round(torch.cuda.max_memory_allocated(device) / 2 ** 30, 2)
            st["peak_reserved_gib"] = round(torch.cuda.max_memory_reserved(device) / 2 ** 30, 2)
        stats["sets"][set_name] = st
    with open(os.path.join(E.W, "parts", f"{tag}_r{rank}.stats.json"), "w") as fh:
        json.dump(stats, fh, indent=1)


def done_keys(set_name):
    ks = set()
    for f in glob.glob(os.path.join(E.W, "parts", set_name, "*.npz")):
        with np.load(f) as z:
            ks.update(z["keys"].astype(str).tolist())
    return ks


def cmd_embed(sets):
    assert E.md5(os.path.join(ENCODE_DIR, "encode.py")) == ENCODE_MD5
    n_dev = torch.cuda.device_count()
    devices = [f"cuda:{i}" for i in range(n_dev)] or ["cpu"]
    deadline = float(os.environ.get("DEADLINE", "0"))
    tag = os.environ.get("SLURM_JOB_ID", "local") + time.strftime("_%H%M%S")
    per_rank = [[] for _ in devices]
    for set_name in sets:
        recs, uniq = unique_items(set_name)
        done = done_keys(set_name)
        todo = [(k, s) for k, s in uniq.items() if k not in done]
        E.log(f"{set_name}: {len(recs)} records, {len(uniq)} unique, {len(done)} done, {len(todo)} to embed "
              f"({sum(len(s) for _, s in todo):,} residues)")
        if not todo:
            continue
        batches = E.S3.make_batches(todo, E.MAX_TOKENS, E.MAX_BATCH, E.MAX_PAIRS)
        shards = E.S3.assign_batches(batches, len(devices))
        for r in range(len(devices)):
            per_rank[r].append((set_name, sorted(shards[r], key=lambda b: -len(b[0][1]))))
        E.log(f"{set_name}: {len(batches)} batches over {len(devices)} device(s)")
    if not any(per_rank):
        return
    t0 = time.time()
    if len(devices) == 1:
        worker(0, devices[0], per_rank[0], tag, deadline)
    else:
        ctx = mp.get_context("spawn")
        ps = [ctx.Process(target=worker, args=(r, d, per_rank[r], tag, deadline)) for r, d in enumerate(devices)]
        for p in ps:
            p.start()
        for p in ps:
            p.join()
        bad = [r for r, p in enumerate(ps) if p.exitcode != 0]
        if bad:
            raise SystemExit(f"ranks {bad} failed")
    E.log(f"embed wall {time.time() - t0:.1f} s")


def cmd_merge(sets):
    import esm
    st = torch.load(E.MODEL, map_location="cpu")
    for set_name in sets:
        recs, uniq = unique_items(set_name)
        got, info = {}, {}
        for f in sorted(glob.glob(os.path.join(E.W, "parts", set_name, "*.npz"))):
            with np.load(f) as z:
                ks, Ls = z["keys"].astype(str), z["lengths"]
                off = np.concatenate([[0], np.cumsum(Ls)])
                assert off[-1] == len(z["q"]) == len(z["codes"]), f
                Q, Cc = z["q"], z["codes"]
                for i, k in enumerate(ks):
                    got[k] = (Q[off[i]:off[i + 1]], Cc[off[i]:off[i + 1]])
                    info[k] = (float(z["maxabs"][i]), int(z["tok_ok"][i]), int(z["unk"][i]))
        missing = [k for k in uniq if k not in got]
        if missing:
            E.log(f"{set_name}: {len(missing)} of {len(uniq)} not embedded yet -> not merged")
            continue
        ids = [h.split()[0] for h, _ in recs]
        assert len(set(ids)) == len(ids)
        seqs = [s for _, s in recs]
        lengths = np.array([len(s) for s in seqs], np.int64)
        offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        Q = np.concatenate([got[E.key_of(s)][0] for s in seqs])
        Cc = np.concatenate([got[E.key_of(s)][1] for s in seqs])
        assert len(Q) == len(Cc) == offsets[-1]
        maxabs = np.array([info[E.key_of(s)][0] for s in seqs])
        tok_ok = np.array([info[E.key_of(s)][1] for s in seqs])
        unk = np.array([info[E.key_of(s)][2] for s in seqs])
        stats = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(E.W, "parts", "*.stats.json")))]
        meta = {"set": set_name, "run": E.RUN, "model_path": E.MODEL, "model_md5": E.md5(E.MODEL),
                "cfg": st["cfg"], "d": st["cfg"]["d"], "fasta": SETS[set_name], "fasta_md5": E.md5(SETS[set_name]),
                "n_records": len(recs), "n_unique_sequences": len(uniq), "n_residues": int(offsets[-1]),
                "n_token_length_mismatch": int((tok_ok == 0).sum()), "n_unk_tokens": int(unk.sum()),
                "max_abs_embedding": float(maxabs.max()), "encode_py_md5": E.md5(os.path.join(ENCODE_DIR, "encode.py")),
                "embedding": {"script": os.path.join(E.REPO, "src/scripts/esm3_sequence_embeddings.py"),
                              "script_md5": E.md5(os.path.join(E.REPO, "src/scripts/esm3_sequence_embeddings.py")),
                              "model": "esm3_sm_open_v1", "esm_version": getattr(esm, "__version__", None),
                              "torch": torch.__version__, "precision": E.PRECISION, "max_tokens": E.MAX_TOKENS,
                              "max_batch": E.MAX_BATCH, "max_pairs": E.MAX_PAIRS},
                "input_transform": "zcw50_clean (" + os.path.join(E.EMB, "clean_pairs", "zcw50_clean.npz") + ", md5 "
                                   + E.md5(os.path.join(E.EMB, "clean_pairs", "zcw50_clean.npz")) + ")",
                "stored": "q fp16 [N_res, 64] = f_q(xw); codes uint8 [N_res, 8] = packbits(f_t(xw) >= 0)",
                "rank_stats": stats}
        hb = [h.encode("utf-8") for h, _ in recs]
        blob = {"offsets": offsets, "ids": np.array(ids), "lengths": lengths, "q": Q, "codes": Cc,
                "seq_concat": np.frombuffer("".join(seqs).encode("ascii"), np.uint8),
                "header_concat": np.frombuffer(b"".join(hb), np.uint8),
                "header_offsets": np.concatenate([[0], np.cumsum([len(b) for b in hb])]).astype(np.int64),
                "meta_json": np.array(json.dumps(meta))}
        path = out_path(set_name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + ".tmp", "wb") as fh:
            np.savez(fh, **blob)
        os.replace(path + ".tmp", path)
        E.log(f"merged {set_name}: {len(recs)} records, {offsets[-1]:,} residues -> {path} md5 {E.md5(path)}; "
              f"tok mismatches {meta['n_token_length_mismatch']}, unk {meta['n_unk_tokens']}, "
              f"max |emb| {meta['max_abs_embedding']:.0f}")


def compare(a, b, d=64):
    b1 = np.unpackbits(a, axis=1)[:, :d]
    b0 = np.unpackbits(b, axis=1)[:, :d]
    return int((b1 == b0).sum()), int(b1.size)


def cmd_verify():
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    E.fp_setup()
    t0 = time.time()
    embedder = E.S3.SequenceOnlyESM3(dev, precision=E.PRECISION)
    enc = E.Enc(dev)
    out = {"device": str(dev), "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else None,
           "gpu_total_gib": (torch.cuda.get_device_properties(dev).total_memory / 2 ** 30) if dev.type == "cuda" else None,
           "torch": torch.__version__}

    def reencode(items):
        batches = E.S3.make_batches(items, E.MAX_TOKENS, E.MAX_BATCH, E.MAX_PAIRS)
        got, emb = {}, {}
        for batch in batches:
            embs = E.embed_batch(embedder, [s for _, s in batch])
            _, c = enc(np.concatenate(embs))
            o = 0
            for (k, s), e in zip(batch, embs):
                got[k] = c[o:o + len(s)]
                emb[k] = e
                o += len(s)
        return got, emb, len(batches)

    zt = E.load_set(BENCH_T)
    pick = np.sort(np.random.default_rng(20260927).choice(len(zt["ids"]), 20, replace=False))
    items = [(E.key_of(zt["sequences"][i]), zt["sequences"][i]) for i in pick]
    got, _, nb = reencode(items)
    eq = tot = ident = 0
    for i, (k, s) in zip(pick, items):
        a, b = compare(got[k], zt["codes"][zt["offsets"][i]:zt["offsets"][i + 1]])
        eq += a; tot += b; ident += int(a == b)
    out["a_benchmark_targets"] = dict(n=len(pick), batches=nb, bit_agreement=eq / tot, items_identical=ident)
    E.log("(a)", out["a_benchmark_targets"])

    zq = np.load(out_path("long_q"))
    L = zq["lengths"]
    seqs = zq["seq_concat"].tobytes().decode("ascii")
    off = zq["offsets"]
    rng = np.random.default_rng(7)
    pick2 = []
    for lo, hi in ((1000, 1500), (1500, 2000), (2000, 2500), (2500, 3001)):
        idx = np.flatnonzero((L >= lo) & (L < hi))
        pick2.extend(sorted(rng.choice(idx, 3, replace=False).tolist()))
    items2 = [(E.key_of(seqs[off[i]:off[i + 1]]), seqs[off[i]:off[i + 1]]) for i in pick2]
    got2, emb2, nb2 = reencode(items2)
    per = []
    for i, (k, s) in zip(pick2, items2):
        prod = zq["codes"][off[i]:off[i + 1]]
        a, b = compare(got2[k], prod)
        rec = dict(row=int(i), len=len(s), rerun_bits_equal=a, bits=b, rerun_identical=a == b)
        if dev.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(dev)
        t1 = time.time()
        try:
            ref = embedder.reference(s).numpy()
            rec["reference_s"] = round(time.time() - t1, 2)
            if dev.type == "cuda":
                rec["reference_peak_alloc_gib"] = round(torch.cuda.max_memory_allocated(dev) / 2 ** 30, 2)
            x, y = ref.astype(np.float64), emb2[k].astype(np.float64)
            cos = (x * y).sum(1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1))
            _, cref = enc(np.ascontiguousarray(ref.astype(np.float16)))
            a2, b2 = compare(cref, prod)
            rec.update(reference_vs_batched_cos_min=float(cos.min()), reference_vs_batched_cos_mean=float(cos.mean()),
                       reference_codes_bits_equal=a2, reference_code_bit_agreement=a2 / b2)
        except torch.cuda.OutOfMemoryError as ex:
            rec["reference_error"] = "OOM: " + str(ex).split("\n")[0][:300]
            rec["reference_s"] = round(time.time() - t1, 2)
            if dev.type == "cuda":
                rec["reference_peak_alloc_gib_before_oom"] = round(torch.cuda.max_memory_allocated(dev) / 2 ** 30, 2)
            torch.cuda.empty_cache()
        per.append(rec)
        E.log("(b)", rec)
    ok = [p for p in per if "reference_error" not in p]
    out["b_long"] = dict(n=len(per), batches=nb2, rerun_bit_agreement=sum(p["rerun_bits_equal"] for p in per) / sum(p["bits"] for p in per),
                         rerun_items_identical=sum(p["rerun_identical"] for p in per),
                         reference_ok=len(ok), reference_oom=len(per) - len(ok),
                         reference_code_bit_agreement=(sum(p["reference_codes_bits_equal"] for p in ok) /
                                                       max(sum(p["bits"] for p in ok), 1)),
                         per_item=per)
    out["wall_s"] = round(time.time() - t0, 1)
    os.makedirs(os.path.join(E.W, "results"), exist_ok=True)
    with open(os.path.join(E.W, "results", f"verify_long.{os.environ.get('SLURM_JOB_ID', 'local')}.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "embed":
        cmd_embed(sys.argv[2:])
        cmd_merge(sys.argv[2:])
    elif cmd == "merge":
        cmd_merge(sys.argv[2:])
    elif cmd == "verify":
        cmd_verify()
    else:
        raise SystemExit(__doc__)
