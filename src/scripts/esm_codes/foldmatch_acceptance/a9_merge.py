#!/usr/bin/env python
"""P8 A9 (long sequences), step 4: the production scores in the P0.7 layout, and how they compare with the research
scores. NERSC CPU node (light: about 10 GB, minutes).

Reads the a9_stage2.py parts (A9/stage2/<task>_p<k>of<K>/scores.npz, every stage2.json 'ok') and writes
  A9/prod_out/        lq_pan_{fwd,rev}.npy [3,000 x 1,000]   production panel S_f / S_r of the long queries (from the
                                                              run: analyze_long's own np.quantile gives their Q90)
                      lq_ben_{fwd,rev}.npy [3,000 x 30,000]  long queries x DB_BEN
                      nq_long_{fwd,rev}.npy [1,465 x 4,000]  benchmark queries x DB_LONG
                      index.npz (the research index arrays, rebuilt from the production inputs and checked equal),
                      manifest.json (md5 of every file, which analyze_long.main() checks), bench_q90_prod.npy (the
                      production Q90 of the 1,465 benchmark queries)
  A9/prod_bscore/out/ S_D_{fwd,rev}.npy [1,465 x 50,000]: production nq_ben in the lq_ben columns (ben_cols), NaN
                      elsewhere (analyze_long uses only those columns); index.npz = the research benchmark index
                      (query / target order and lengths)
  A9/merge.json       checks and the production-vs-research comparison (informative; A4 is the parity gate):
                      code bits of the 34,000 database targets (production build vs research encode), the query
                      encodings q, S, Q90 and |delta ln E| = |lambda(m) (dS - dQ90)| on pairs with E <= 1
Usage:  python a9_merge.py
"""
from __future__ import annotations

import glob
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import acc_common as C  # noqa: E402  (first: it sets the BLAS thread variables before numpy loads)
import numpy as np  # noqa: E402


def load_task(A9, task):
    dirs = sorted(glob.glob(os.path.join(A9, "stage2", f"{task}_p*of*")))
    assert dirs, task
    nparts = {int(d.rsplit("of", 1)[1]) for d in dirs}
    assert len(nparts) == 1 and len(dirs) == nparts.pop(), f"{task}: parts {dirs}"
    recs, zs = [], []
    for d in dirs:
        r = C.read_json(os.path.join(d, "stage2.json"))
        assert r["ok"], f"{d}: stage2 not ok"
        assert C.sha256_file(os.path.join(d, "scores.npz")) == r["files"]["scores_npz"], d
        recs.append(dict(dir=d, stage2_json_sha256=C.sha256_file(os.path.join(d, "stage2.json")),
                         scores_npz_sha256=r["files"]["scores_npz"], queries=r["queries"], N_db=r["N_db"],
                         system=r["system"], max_evalue=r["max_evalue"]))
        zs.append(np.load(os.path.join(d, "scores.npz")))
    rows = np.concatenate([z["qrows"] for z in zs])
    nq = len(rows)
    assert np.array_equal(np.sort(rows), np.arange(nq)), f"{task}: parts do not cover the queries once"
    out = {}
    for k in ("S_f", "S_r", "lnE", "emb", "panel_f", "panel_r"):
        a0 = zs[0][k]
        M = np.empty((nq,) + a0.shape[1:], a0.dtype)
        for z in zs:
            M[z["qrows"]] = z[k]
        out[k] = M
    out["q90"] = np.empty(nq)
    for z in zs:
        out["q90"][z["qrows"]] = z["q90"]
    out["col_ids"] = zs[0]["col_ids"].astype(str)
    return out, recs


def bits_agreement(prod, ref):
    x = np.unpackbits(np.bitwise_xor(prod, ref), axis=1)
    return 1.0 - x.mean(), x.mean(axis=1)


def main():
    from foldmatch.search import scoring_system as ss
    from foldmatch.search.residue_code_store import ResidueCodeStore
    A9 = C.result_dir("A9")
    inp = np.load(os.path.join(A9, "inputs", "a9_inputs.npz"))
    rix = np.load(C.inp("p0/long/out/index.npz"))
    rec = {"title": "P8 A9: production scores in the P0.7 layout; production vs research", "parts": {}}
    T = {}
    for task in ("nq_long", "nq_ben", "lq_ben"):
        T[task], rec["parts"][task] = load_task(A9, task)
    systems = {json_key["system"]["digest"] for v in rec["parts"].values() for json_key in v}
    assert len(systems) == 1, systems
    # production panel order = the research shipped panel order?
    prep = ss.prepare(ss.get(ss.DEFAULT_ESM_CODES), local_files_only=True)
    pcodes, poff = prep.for_worker().load_panel()
    zp = np.load(C.inp("embcomp/panel/fit2/results/shipped_panel_codes.npz"))
    panel_same = bool(np.array_equal(pcodes, zp["codes"]) and np.array_equal(poff, zp["offsets"]))
    assert panel_same, "the production panel is not the shipped research panel in its order"
    nq_long, nq_ben, lq_ben = T["nq_long"], T["nq_ben"], T["lq_ben"]
    chk = dict(panel_is_shipped_panel_in_order=panel_same,
               nq_q90_same_in_both_runs=bool(np.array_equal(nq_long["q90"], nq_ben["q90"])),
               nq_panel_scores_same_in_both_runs=bool(np.array_equal(nq_long["panel_f"], nq_ben["panel_f"])
                                                      and np.array_equal(nq_long["panel_r"], nq_ben["panel_r"])),
               lq_q90_eq_np_quantile=bool(np.array_equal(lq_ben["q90"], C.q90_of(
                   lq_ben["panel_f"].astype(np.float64) - lq_ben["panel_r"].astype(np.float64)))),
               columns=dict(nq_long=bool(np.array_equal(nq_long["col_ids"], rix["long_ids"].astype(str))),
                            lq_ben=bool(np.array_equal(lq_ben["col_ids"], inp["db_ben_ids"].astype(str))),
                            nq_ben=bool(np.array_equal(nq_ben["col_ids"], inp["db_ben_ids"].astype(str)))))
    assert chk["nq_q90_same_in_both_runs"] and chk["lq_q90_eq_np_quantile"] and all(chk["columns"].values()), chk

    # ---------------------------------------------------------------- prod_out (the research out/ layout)
    po = C.result_dir("A9", "prod_out")
    files = {}

    def save(name, arr):
        p = os.path.join(po, name)
        np.save(p, arr)
        files[name] = C.md5_file(p)
    save("lq_pan_fwd.npy", lq_ben["panel_f"])
    save("lq_pan_rev.npy", lq_ben["panel_r"])
    save("lq_ben_fwd.npy", lq_ben["S_f"])
    save("lq_ben_rev.npy", lq_ben["S_r"])
    save("nq_long_fwd.npy", nq_long["S_f"])
    save("nq_long_rev.npy", nq_long["S_r"])
    long_is_rev = np.array([x.startswith("lrev_") for x in inp["db_long_ids"].astype(str)])
    ix = dict(lq_ids=inp["lq_ids"], lq_len=inp["lq_len"], nq_ids=inp["nq_ids"], nq_len=inp["nq_len"],
              pan_ids=zp["ids"], pan_len=np.diff(poff), ben_cols=inp["ben_cols"], ben_len=inp["db_ben_len"],
              long_ids=inp["db_long_ids"], long_len=inp["db_long_len"], long_is_rev=long_is_rev)
    for k, v in ix.items():
        x, y = np.asarray(v), rix[k]
        if y.dtype.kind == "U":
            x, y = x.astype(str), y.astype(str)
        assert np.array_equal(x, y), f"index.npz {k}: production inputs differ from the research index"
    np.savez(os.path.join(po, "index.npz"), **{k: rix[k] for k in rix.files})   # identical arrays, research dtypes
    files["index.npz"] = C.md5_file(os.path.join(po, "index.npz"))
    np.save(os.path.join(po, "bench_q90_prod.npy"), nq_long["q90"])
    man = dict(source="FoldMatch cc55c01 production path (a9_qpass.py + fm-search build sequences --residue-codes + "
                      "a9_stage2.py); written by a9_merge.py", setting=["D", 1.0, 5.0, 0.5], runs=["D_fwd", "D_rev"],
               rev_sub=5000, seed=20260927, parts=rec["parts"], files=files)
    C.write_json(man, os.path.join(po, "manifest.json"))
    pb = C.result_dir("A9", "prod_bscore", "out")
    bix = np.load(C.inp("embcomp/evalue/score/out/index.npz"))
    nqn, ntb = len(bix["query_ids"]), len(bix["target_ids"])
    for d, k in (("fwd", "S_f"), ("rev", "S_r")):
        M = np.full((nqn, ntb), np.nan, np.float32)
        M[:, inp["ben_cols"]] = nq_ben[k]
        np.save(os.path.join(pb, f"S_D_{d}.npy"), M)
    dst = os.path.join(pb, "index.npz")
    if not os.path.lexists(dst):
        os.symlink(os.path.realpath(C.inp("embcomp/evalue/score/out/index.npz")), dst)
    rec["prod_out"] = dict(dir=po, files=files, manifest_sha256=C.sha256_file(os.path.join(po, "manifest.json")),
                           bench_q90_prod_md5=C.md5_file(os.path.join(po, "bench_q90_prod.npy")),
                           prod_bscore={f: C.md5_file(os.path.join(pb, f)) for f in ("S_D_fwd.npy", "S_D_rev.npy")})

    # ---------------------------------------------------------------- production vs research (informative)
    cmp = {}
    # code bits: the two production databases vs the research encodings
    stores = {"long": ResidueCodeStore.from_db_path(os.path.join(A9, "db/long/L")),
              "ben": ResidueCodeStore.from_db_path(os.path.join(A9, "db/ben/B"))}
    ref_long = {k: np.load(C.inp(f"p0/long/enc/out/long_{k}.L2m_B8_s0.npz")) for k in ("t", "r")}
    ref_bt = np.load(C.inp("embcomp/evalue/encode/out/targets.L2m_B8_s0.npz"))
    for name, ids, refs in (("db_long", inp["db_long_ids"].astype(str), [ref_long["t"], ref_long["r"]]),
                            ("db_ben", inp["db_ben_ids"].astype(str), [ref_bt])):
        store = "long" if name == "db_long" else "ben"
        codes = stores[store].open_memmap()
        offs, lens = stores[store].lookup_rows(ids.tolist())
        rpos = {}
        for z in refs:
            for j, t in enumerate(z["ids"].astype(str).tolist()):
                rpos[t] = (z, j)
        prod_rows, ref_rows, item = [], [], []
        for t, o, m in zip(ids.tolist(), offs.tolist(), lens.tolist()):
            z, j = rpos[t]
            ro, rl = int(z["offsets"][j]), int(z["lengths"][j])
            assert rl == m, t
            prod_rows.append(np.asarray(codes[o:o + m]))
            ref_rows.append(z["codes"][ro:ro + m])
            item.append(m)
        P, Rr = np.concatenate(prod_rows), np.concatenate(ref_rows)
        agree, per_res = bits_agreement(P, Rr)
        cut = np.cumsum([0] + item)
        per_item = np.array([1.0 - per_res[cut[k]:cut[k + 1]].mean() for k in range(len(item))])
        L = np.array(item)
        bins = {}
        for lo, hi in ((0, 300), (300, 1000), (1000, 1500), (1500, 2000), (2000, 2500), (2500, 3001)):
            msk = np.repeat((L >= lo) & (L < hi), L)
            if msk.any():
                bins[f"{lo}-{hi - 1}"] = float(1.0 - per_res[msk].mean())
        cmp[f"code_bits_{name}"] = dict(bits=int(P.size * 8), agreement=float(agree), by_length=bins,
                                        worst_item=float(per_item.min()), items_identical=int((per_item == 1).sum()),
                                        items=len(item))
    # query encodings q: production query pass vs the research encodings
    for name, zref in (("lq", np.load(C.inp("p0/long/enc/out/long_q.L2m_B8_s0.npz"))),
                       ("nq", np.load(C.inp("embcomp/evalue/encode/out/queries.L2m_B8_s0.npz")))):
        ent = C.read_json(os.path.join(A9, "qpass", name, "entries.json"))
        mm = np.memmap(ent["path"], dtype=np.float16, mode="r", shape=(ent["rows"], ent["width"]))
        cos, same, n = [], 0, 0
        for j, q in enumerate(zref["ids"].astype(str).tolist()):
            o, m, _ = ent["entries"][q]
            a = np.asarray(mm[o:o + m], np.float32)
            b = zref["q"][int(zref["offsets"][j]):int(zref["offsets"][j + 1])].astype(np.float32)
            cos.append((a * b).sum(1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-30))
            same += int((a == b).sum())
            n += a.size
        cos = np.concatenate(cos)
        cmp[f"q_{name}"] = dict(residues=int(len(cos)), cosine_median=float(np.median(cos)),
                                cosine_p01=float(np.percentile(cos, 1)), cosine_min=float(cos.min()),
                                fp16_values_identical=same / n)
    # S, Q90 and ln E against the research matrices
    tab = C.ship_tables()["all"]
    q90_res_nq = np.load(C.inp("embcomp/panel/fit2/results/bench_q90_shipped.npy"))
    rS = lambda n: np.load(os.path.join(C.inp("p0/long/out"), n)).astype(np.float64)  # noqa: E731
    q90_res_lq = C.q90_of(rS("lq_pan_fwd.npy") - rS("lq_pan_rev.npy"))
    Sb_f = np.load(C.inp("embcomp/evalue/score/out/S_D_fwd.npy"), mmap_mode="r")
    Sb_r = np.load(C.inp("embcomp/evalue/score/out/S_D_rev.npy"), mmap_mode="r")
    res_nq_ben = (np.asarray(Sb_f[:, inp["ben_cols"]], np.float64) - np.asarray(Sb_r[:, inp["ben_cols"]], np.float64))
    for name, prod, Sres, q90p, q90r, m, N in (
            ("nq_long", nq_long, rS("nq_long_fwd.npy") - rS("nq_long_rev.npy"), nq_long["q90"], q90_res_nq,
             inp["nq_len"], float(inp["db_long_len"].sum())),
            ("nq_ben", nq_ben, res_nq_ben, nq_ben["q90"], q90_res_nq, inp["nq_len"], float(inp["db_ben_len"].sum())),
            ("lq_ben", lq_ben, rS("lq_ben_fwd.npy") - rS("lq_ben_rev.npy"), lq_ben["q90"], q90_res_lq,
             inp["lq_len"], float(inp["db_ben_len"].sum()))):
        Sp = prod["S_f"].astype(np.float64) - prod["S_r"].astype(np.float64)
        dS = np.abs(Sp - Sres)
        Lp = C.ln_evalue(tab, m, Sp, q90p, N)
        Lr = C.ln_evalue(tab, m, Sres, q90r, N)
        sel = (Lp <= 0.0) | (Lr <= 0.0)
        dL = np.abs(Lp - Lr)[sel]
        dq = np.abs(q90p - q90r)
        cmp[f"scores_{name}"] = dict(
            pairs=int(Sp.size), dS_median=float(np.median(dS)), dS_p95=float(np.percentile(dS, 95)),
            dS_max=float(dS.max()), dQ90_median=float(np.median(dq)), dQ90_max=float(dq.max()),
            pairs_E_le_1_either=int(sel.sum()), dlnE_median=float(np.median(dL)) if dL.size else None,
            dlnE_p95=float(np.percentile(dL, 95)) if dL.size else None, dlnE_max=float(dL.max()) if dL.size else None,
            N=N, note="E at N = the database residues, shipped 'all' table; pairs with E <= 1 in either")
        if name == "lq_ben":
            for lab, msk in (("queries_le_966", m <= 966), ("queries_gt_966", m > 966)):
                s2 = sel[msk]
                d2 = np.abs(Lp[msk] - Lr[msk])[s2]
                cmp[f"scores_{name}"][lab] = dict(pairs_E_le_1_either=int(s2.sum()),
                                                  dlnE_median=float(np.median(d2)) if d2.size else None,
                                                  dlnE_p95=float(np.percentile(d2, 95)) if d2.size else None,
                                                  dQ90_median=float(np.median(dq[msk])))
    rec["checks"] = chk
    rec["production_vs_research"] = cmp
    rec["a4_thresholds_for_reference"] = "A4 gates (A4_THRESHOLDS.json): code bits >= 0.990, |dlnE| median <= 0.1, " \
                                         "P95 <= 0.5 (on P_sub hits); here informative only"
    rec["provenance"] = C.provenance(kernel=False)
    sha = C.write_json(rec, os.path.join(A9, "merge.json"))
    C.log("merge", chk, {k: (v.get("agreement") or v.get("cosine_median") or v.get("dlnE_median"))
                         for k, v in cmp.items()}, "sha256", sha)


if __name__ == "__main__":
    main()
