#!/usr/bin/env python
"""P0.4 spike inputs (run once, on the Mac; the npz files are then copied to NERSC so every build reads the same bytes).

golden_inputs.npz  1,000 seeded synthetic pairs for the cross-build golden hash (q fp16 [n, 64], codes uint8 [m, 8]):
                   500 random (n, m in [1, 400]), 400 'related' (target codes = sign bits of a noisy, shifted, gapped copy
                   of the query rows, so alignments are long and gapped), 100 edge cases (n = 1, m = 1, all-identical
                   codes, all-identical query rows, zero query rows, one-row both).
bench_data.npz     the research throughput workload (embcomp/throughput/bench/data/workloads.pkl, seed 20260925): real
                   query encodings q16 for Lq 100 / 300 / 600 / 1000 and the real 8-B codes of the first 1,000 (and 10,000)
                   chain-set candidates (with replacement from the 559 evaluation chains).
usage: make_inputs.py OUT_DIR WORKLOADS_PKL
"""
import os
import pickle
import sys

import numpy as np


def golden(rng):
    qs, cs, kinds = [], [], []

    def add(q, c, kind):
        qs.append(np.asarray(q, np.float16))
        cs.append(np.asarray(c, np.uint8))
        kinds.append(kind)

    for _ in range(500):
        n, m = int(rng.integers(1, 401)), int(rng.integers(1, 401))
        add(rng.standard_normal((n, 64)), rng.integers(0, 256, (m, 8)), "random")
    for _ in range(400):
        n = int(rng.integers(20, 401))
        q = rng.standard_normal((n, 64)).astype(np.float32)
        a, b = sorted(rng.integers(0, n, 2))
        seg = q[a:b + 1]
        keep = rng.random(len(seg)) > 0.1                                    # deletions
        seg = seg[keep]
        ins = rng.standard_normal((int(rng.integers(0, 10)), 64)).astype(np.float32)
        pos = int(rng.integers(0, len(seg) + 1))
        seg = np.concatenate([seg[:pos], ins, seg[pos:]])                    # an insertion block
        pre = rng.standard_normal((int(rng.integers(0, 60)), 64)).astype(np.float32)
        post = rng.standard_normal((int(rng.integers(0, 60)), 64)).astype(np.float32)
        t = np.concatenate([pre, seg, post])
        if len(t) == 0:
            t = rng.standard_normal((1, 64)).astype(np.float32)
        t = t + float(rng.uniform(0.3, 1.5)) * rng.standard_normal(t.shape).astype(np.float32)
        add(q, np.packbits(t >= 0, axis=1), "related")
    for e in range(100):
        k = e % 6
        n, m = int(rng.integers(1, 200)), int(rng.integers(1, 200))
        if k == 0:
            q, c = rng.standard_normal((1, 64)), rng.integers(0, 256, (m, 8))
        elif k == 1:
            q, c = rng.standard_normal((n, 64)), rng.integers(0, 256, (1, 8))
        elif k == 2:
            q, c = rng.standard_normal((n, 64)), np.repeat(rng.integers(0, 256, (1, 8)), m, axis=0)
        elif k == 3:
            q, c = np.repeat(rng.standard_normal((1, 64)), n, axis=0), rng.integers(0, 256, (m, 8))
        elif k == 4:
            q = rng.standard_normal((n, 64))
            q[rng.random(n) < 0.3] = 0.0
            c = rng.integers(0, 256, (m, 8))
        else:
            q, c = rng.standard_normal((1, 64)), rng.integers(0, 256, (1, 8))
        add(q, c, f"edge{k}")
    qo = np.concatenate([[0], np.cumsum([len(x) for x in qs])]).astype(np.int64)
    co = np.concatenate([[0], np.cumsum([len(x) for x in cs])]).astype(np.int64)
    return dict(q=np.concatenate(qs), q_off=qo, codes=np.concatenate(cs), c_off=co, kind=np.array(kinds))


def bench(pkl):
    with open(pkl, "rb") as f:
        W = pickle.load(f)
    out = dict(seed=np.int64(W["seed"]))
    for Lq in (100, 300, 600, 1000):
        out[f"q16_{Lq}"] = np.ascontiguousarray(W["queries"][Lq]["q16"], np.float16)
    pool = W["pool"]
    for K in (1000, 10000):
        cand = W["cand"][:K]
        codes = [pool[c["pool_index"]]["codes8"] for c in cand]
        out[f"codes_K{K}"] = np.ascontiguousarray(np.concatenate(codes), np.uint8)
        out[f"off_K{K}"] = np.concatenate([[0], np.cumsum([len(c) for c in codes])]).astype(np.int64)
    return out


def main():
    od, pkl = sys.argv[1], sys.argv[2]
    os.makedirs(od, exist_ok=True)
    g = golden(np.random.default_rng(20260927))
    np.savez(os.path.join(od, "golden_inputs.npz"), **g)
    b = bench(pkl)
    np.savez(os.path.join(od, "bench_data.npz"), **b)
    print("golden pairs", len(g["q_off"]) - 1, "q rows", len(g["q"]), "code rows", len(g["codes"]))
    print("bench", {k: v.shape for k, v in b.items()})


if __name__ == "__main__":
    main()
