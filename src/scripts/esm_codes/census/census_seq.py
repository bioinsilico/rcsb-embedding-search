#!/usr/bin/env python
"""P0.6 degenerate / low-information census, step 1: SEQUENCE LEVEL (PDB seqres + UniRef50) and the item selection for
the code-level step.

Sources
  pdb      pdb_seqres.txt.gz (wwPDB derived_data, 67,128,902 B, 2026-09-19); only 'mol:protein' records are counted.
  uniref50 /pscratch/sd/j/jsegura/uniref50/uniref50.fasta (release 2026_03, 38,840,027 cluster representatives).

Per record (and per unique sequence) the requested categories (L = length, letters as given; nothing normalised):
  L1        L == 1
  Llt21     L < 21                                             (includes L1)
  allX      every residue is 'X'                               (PDB seqres writes UNK / unmapped residues as X)
  homopoly  one distinct letter, L >= 2, letter != 'X'         (all-X is its own category)
  histag    His-tag-dominated: a run of >= 6 consecutive 'H' AND H count / L >= 0.25
  nonstd    > 1% of residues outside the 20 standard letters   (S13 / S15 definition: anything but ACDEFGHIKLMNPQRSTVWY)
Informational (not D14 categories): hisrun6 (any run of >= 6 H), lowcx (<= 3 distinct letters, L >= 21, not homopoly /
allX), Lgt1000 (> 1,000: outside the S13 target domain), Lgt966 (> 966: outside the S13 query domain).

Item selection for the code-level step (per source, per requested category, on UNIQUE sequences with L <= LMAX_ENC):
all items if <= CAP, else a seeded uniform sample of CAP; plus RANDOM = 2,000 uniform unique sequences per source
(pdb: uniform over unique protein sequences, L <= LMAX_ENC; uniref50: keyed-hash pre-sample of ~2x, then a seeded
uniform draw of 2,000 among those with L <= LMAX_ENC).  Items are deduplicated across sources by exact sequence.

Outputs (OUT dir): seq_census.json, items.tsv (item, source, category memberships, length, sequence), items.fasta.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import time
from multiprocessing import Pool

import numpy as np

STD = b"ACDEFGHIKLMNPQRSTVWY"
ISSTD = np.zeros(256, bool)
for _c in STD:
    ISSTD[_c] = True
CATS = ["L1", "Llt21", "allX", "homopoly", "histag", "nonstd"]
INFO = ["hisrun6", "lowcx", "Lgt1000", "Lgt966"]
ALLC = CATS + INFO
HRUN = re.compile(rb"H{6,}")
SEED = 20260927
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def features(s: bytes):
    """category bitmask (ALLC order) and a few numbers for one sequence."""
    L = len(s)
    a = np.frombuffer(s, np.uint8)
    nstd = int(ISSTD[a].sum())
    nX = s.count(b"X")
    distinct = len(set(s))
    nH = s.count(b"H")
    hrun = max((len(m) for m in HRUN.findall(s)), default=0) if b"HHHHHH" in s else 0
    f = 0
    if L == 1:
        f |= 1 << 0
    if L < 21:
        f |= 1 << 1
    if L >= 1 and nX == L:
        f |= 1 << 2
    homo = L >= 2 and distinct == 1 and s[:1] != b"X"
    if homo:
        f |= 1 << 3
    if hrun >= 6 and nH >= 0.25 * L:
        f |= 1 << 4
    if L > 0 and (L - nstd) > 0.01 * L:
        f |= 1 << 5
    if hrun >= 6:
        f |= 1 << 6
    if L >= 21 and distinct <= 3 and not homo and nX != L:
        f |= 1 << 7
    if L > 1000:
        f |= 1 << 8
    if L > 966:
        f |= 1 << 9
    return f, L, L - nstd


def keyed_hash(cid: bytes) -> int:
    return int.from_bytes(hashlib.blake2b(cid, digest_size=8, key=b"p0-census-random").digest(), "big")


# ------------------------------------------------------------------------------------------------ UniRef50 (parallel)
HASH_P = None   # set in main (probability of the hash pre-sample)


def process_block(args):
    bid, block, hash_thr = args
    recs = block[1:].split(b"\n>")
    counts = np.zeros(len(ALLC), np.int64)
    n = 0
    nres = 0
    flagged, presample = [], []
    lens = []
    nonstd_letters = {}
    for rec in recs:
        nl = rec.find(b"\n")
        hdr = rec[:nl] if nl >= 0 else rec
        s = rec[nl + 1:].replace(b"\n", b"").strip() if nl >= 0 else b""
        cid = hdr.split(b" ", 1)[0]
        f, L, nns = features(s)
        n += 1
        nres += L
        lens.append(L)
        if nns:
            for ch in set(s) - set(STD):
                nonstd_letters[chr(ch)] = nonstd_letters.get(chr(ch), 0) + s.count(bytes([ch]))
        for k in range(len(ALLC)):
            if f >> k & 1:
                counts[k] += 1
        if f & 0x3F:
            flagged.append((cid.decode(), s.decode(), f))
        if keyed_hash(cid) < hash_thr:
            presample.append((cid.decode(), s.decode(), f))
    return bid, n, nres, counts, flagged, presample, np.bincount(np.minimum(np.array(lens), 40000), minlength=1), \
        nonstd_letters


def blocks(path, bsize, hash_thr):
    with open(path, "rb") as fh:
        buf, k = b"", 0
        while True:
            chunk = fh.read(bsize)
            if not chunk:
                break
            buf += chunk
            cut = buf.rfind(b"\n>")
            if cut < 0:
                continue
            yield k, buf[:cut + 1], hash_thr
            k += 1
            buf = buf[cut + 1:]
        if buf.strip():
            yield k, buf, hash_thr


def scan_uniref50(path, workers, block_mb, expect_records):
    # keyed-hash pre-sample: ~ 3 x 2,000 records (so that >= 2,000 remain after the L <= LMAX_ENC filter)
    hash_thr = int(6000 / expect_records * 2 ** 64)
    counts = np.zeros(len(ALLC), np.int64)
    nrec = nres = 0
    flagged, presample = [], []
    lhist = np.zeros(40001, np.int64)
    letters = {}
    with Pool(workers) as pool:
        for bid, n, r, c, fl, ps, lh, nl in pool.imap(process_block, blocks(path, block_mb << 20, hash_thr), chunksize=1):
            nrec += n
            nres += r
            counts += c
            flagged.extend(fl)
            presample.extend(ps)
            lhist[:len(lh)] += lh
            for k, v in nl.items():
                letters[k] = letters.get(k, 0) + v
            if bid % 100 == 0:
                log(f"uniref50 block {bid}: {nrec:,} records, {len(flagged):,} flagged")
    return dict(nrec=nrec, nres=nres, counts=counts, flagged=flagged, presample=presample, lhist=lhist,
                letters=letters, hash_thr=hash_thr)


# ------------------------------------------------------------------------------------------------ PDB seqres
def scan_pdb(path):
    mol = {}
    recs = []
    with gzip.open(path, "rb") as fh:
        hdr = None
        for line in fh:
            line = line.rstrip(b"\n")
            if line.startswith(b">"):
                hdr = line[1:]
            elif hdr is not None:
                toks = hdr.split()
                m = toks[1].decode() if len(toks) > 1 else "?"
                mol[m] = mol.get(m, 0) + 1
                if m == "mol:protein":
                    recs.append((toks[0].decode(), line.strip()))
                hdr = None
    return mol, recs


# ------------------------------------------------------------------------------------------------ main
def cat_names(f):
    return [c for k, c in enumerate(ALLC) if f >> k & 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", default="/pscratch/sd/j/jsegura/embcomp/p0/census/pdb_seqres.txt.gz")
    ap.add_argument("--uniref50", default="/pscratch/sd/j/jsegura/uniref50/uniref50.fasta")
    ap.add_argument("--out", default="/pscratch/sd/j/jsegura/embcomp/p0/census/seq")
    ap.add_argument("--workers", type=int, default=200)
    ap.add_argument("--block-mb", type=int, default=16)
    ap.add_argument("--cap", type=int, default=2000)
    ap.add_argument("--nrandom", type=int, default=2000)
    ap.add_argument("--lmax-enc", type=int, default=1000)
    ap.add_argument("--uniref50-records", type=int, default=38840027)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(SEED)
    out = {"meta": {"definition": __doc__.strip(), "seed": SEED, "cap_per_category": a.cap, "n_random": a.nrandom,
                    "lmax_encode": a.lmax_enc, "categories": CATS, "informational": INFO,
                    "pdb_file": a.pdb, "pdb_bytes": os.path.getsize(a.pdb),
                    "uniref50_file": a.uniref50, "uniref50_bytes": os.path.getsize(a.uniref50)},
           "sources": {}}
    items = {}   # sequence -> dict(sources={src: set(cats)}, ids={src: [ids]})

    def add_item(seq, src, cats, ident):
        d = items.setdefault(seq, {"cats": {}, "ids": {}})
        d["cats"].setdefault(src, set()).update(cats)
        d["ids"].setdefault(src, []).append(ident)

    # ---------------- PDB
    log("pdb: reading", a.pdb)
    mol, recs = scan_pdb(a.pdb)
    log(f"pdb: mol types {mol}; protein records {len(recs):,}")
    rec_counts = np.zeros(len(ALLC), np.int64)
    uniq = {}
    nres = 0
    letters = {}
    for cid, s in recs:
        f, L, nns = features(s)
        nres += L
        for k in range(len(ALLC)):
            rec_counts[k] += f >> k & 1
        u = uniq.get(s)
        if u is None:
            uniq[s] = [f, 1, cid]
        else:
            u[1] += 1
        if nns:
            for ch in set(s) - set(STD):
                letters[chr(ch)] = letters.get(chr(ch), 0) + s.count(bytes([ch]))
    useqs = list(uniq.keys())
    uf = np.array([uniq[s][0] for s in useqs], np.int64)
    ul = np.array([len(s) for s in useqs], np.int64)
    uniq_counts = [int(((uf >> k) & 1).sum()) for k in range(len(ALLC))]
    L_all = np.array([len(s) for _, s in recs])
    out["sources"]["pdb"] = {
        "mol_types_records": mol, "protein_records": len(recs), "protein_residues": int(nres),
        "unique_protein_sequences": len(useqs), "empty_sequences": int((L_all == 0).sum()),
        "length_quantiles_records": {str(q): float(np.quantile(L_all, q)) for q in (0, .01, .05, .5, .95, .99, 1)},
        "nonstandard_letter_counts": letters,
        "records": {c: int(rec_counts[k]) for k, c in enumerate(ALLC)},
        "unique": {c: uniq_counts[k] for k, c in enumerate(ALLC)},
        "records_any_requested": int(sum(1 for _, s in recs if uniq[s][0] & 0x3F)),
        "unique_any_requested": int(((uf & 0x3F) > 0).sum()),
    }
    # examples (most frequent unique sequences per category)
    ex = {}
    for k, c in enumerate(ALLC[:6]):
        idx = np.flatnonzero((uf >> k) & 1)
        top = sorted(idx, key=lambda i: -uniq[useqs[i]][1])[:8]
        ex[c] = [{"seq": useqs[i].decode()[:80], "L": int(ul[i]), "records": uniq[useqs[i]][1],
                  "first_id": uniq[useqs[i]][2]} for i in top]
    out["sources"]["pdb"]["examples_most_frequent"] = ex
    # selection
    sel_info = {}
    for k, c in enumerate(CATS):
        idx = np.flatnonzero(((uf >> k) & 1).astype(bool) & (ul <= a.lmax_enc) & (ul >= 1))
        n_all = int(((uf >> k) & 1).sum())
        pick = idx if len(idx) <= a.cap else np.sort(rng.choice(idx, a.cap, replace=False))
        for i in pick:
            add_item(useqs[i].decode(), "pdb", [c], uniq[useqs[i]][2])
        sel_info[c] = {"unique": n_all, "unique_L_le_lmax": int(len(idx)), "selected": int(len(pick)),
                       "capped": bool(len(idx) > a.cap)}
    idx = np.flatnonzero((ul <= a.lmax_enc) & (ul >= 1))
    pick = np.sort(rng.choice(idx, a.nrandom, replace=False))
    for i in pick:
        add_item(useqs[i].decode(), "pdb", ["random"], uniq[useqs[i]][2])
    sel_info["random"] = {"pool_unique_L_le_lmax": int(len(idx)), "selected": int(len(pick)),
                          "random_items_in_requested_categories": int(((uf[pick] & 0x3F) > 0).sum())}
    out["sources"]["pdb"]["selection"] = sel_info
    log("pdb done", json.dumps(sel_info))
    del recs, uniq

    # ---------------- UniRef50
    log("uniref50: scanning", a.uniref50)
    r = scan_uniref50(a.uniref50, a.workers, a.block_mb, a.uniref50_records)
    fl = r["flagged"]
    ff = np.array([f for _, _, f in fl], np.int64)
    fls = np.array([len(s) for _, s, _ in fl], np.int64)
    # unique sequences among flagged (UniRef50 representatives are near-unique; dedupe anyway)
    first = {}
    for i, (cid, s, f) in enumerate(fl):
        first.setdefault(s, i)
    ufl = np.array(sorted(first.values()), np.int64)
    lh = r["lhist"]
    cum = np.cumsum(lh)

    def lq(q):
        return int(np.searchsorted(cum, q * cum[-1]))
    ex = {}
    for k, c in enumerate(CATS):
        idx = np.flatnonzero((ff >> k) & 1)[:8]
        ex[c] = [{"seq": fl[i][1][:80], "L": int(fls[i]), "id": fl[i][0]} for i in idx]
    out["sources"]["uniref50"] = {
        "records": int(r["nrec"]), "residues": int(r["nres"]), "nonstandard_letter_counts": r["letters"],
        "length_quantiles_records": {str(q): lq(q) for q in (0.001, .01, .05, .5, .95, .99)},
        "length_min": int(np.flatnonzero(lh)[0]),
        "records_by_category": {c: int(r["counts"][k]) for k, c in enumerate(ALLC)},
        "unique_flagged_sequences": int(len(ufl)), "records_any_requested": int(len(fl)),
        "examples_first_in_file": ex,
        "random_presample": {"hash": "blake2b-64(cluster id, key 'p0-census-random') < thr", "thr": r["hash_thr"],
                             "presampled": len(r["presample"])},
    }
    sel_info = {}
    for k, c in enumerate(CATS):
        idx = ufl[((ff[ufl] >> k) & 1).astype(bool) & (fls[ufl] <= a.lmax_enc) & (fls[ufl] >= 1)]
        n_all = int(((ff[ufl] >> k) & 1).sum())
        pick = idx if len(idx) <= a.cap else np.sort(rng.choice(idx, a.cap, replace=False))
        for i in pick:
            add_item(fl[i][1], "uniref50", [c], fl[i][0])
        sel_info[c] = {"unique": n_all, "unique_L_le_lmax": int(len(idx)), "selected": int(len(pick)),
                       "capped": bool(len(idx) > a.cap)}
    ps = r["presample"]
    pidx = np.array([i for i, (_, s, _) in enumerate(ps) if 1 <= len(s) <= a.lmax_enc])
    pick = np.sort(rng.choice(pidx, a.nrandom, replace=False))
    for i in pick:
        add_item(ps[i][1], "uniref50", ["random"], ps[i][0])
    sel_info["random"] = {"presample_L_le_lmax": int(len(pidx)), "selected": int(len(pick)),
                          "random_items_in_requested_categories": int(sum(1 for i in pick if ps[i][2] & 0x3F))}
    out["sources"]["uniref50"]["selection"] = sel_info
    log("uniref50 done", json.dumps(sel_info))

    # ---------------- items
    seqs = sorted(items.keys(), key=lambda s: (len(s), s))
    with open(os.path.join(a.out, "items.tsv"), "w") as fh, open(os.path.join(a.out, "items.fasta"), "w") as fa:
        fh.write("item\tlength\tpdb_cats\tuniref50_cats\tpdb_ids\tuniref50_ids\tsequence\n")
        for k, s in enumerate(seqs):
            d = items[s]
            pc = ",".join(sorted(d["cats"].get("pdb", [])))
            uc = ",".join(sorted(d["cats"].get("uniref50", [])))
            pid = ",".join(d["ids"].get("pdb", [])[:3])
            uid = ",".join(d["ids"].get("uniref50", [])[:3])
            fh.write(f"it{k:06d}\t{len(s)}\t{pc}\t{uc}\t{pid}\t{uid}\t{s}\n")
            fa.write(f">it{k:06d}\n{s}\n")
    L = np.array([len(s) for s in seqs])
    out["items"] = {"n_items": len(seqs), "residues": int(L.sum()), "max_len": int(L.max()),
                    "per_source_category": {src: {c: sum(1 for s in seqs if c in items[s]["cats"].get(src, ()))
                                                  for c in CATS + ["random"]} for src in ("pdb", "uniref50")},
                    "items_tsv_md5": hashlib.md5(open(os.path.join(a.out, "items.tsv"), "rb").read()).hexdigest()}
    out["meta"]["seconds"] = round(time.time() - T0, 1)
    json.dump(out, open(os.path.join(a.out, "seq_census.json"), "w"), indent=1)
    log("done", json.dumps(out["items"]))


if __name__ == "__main__":
    main()
