"""P9.X NERSC validation: the comparisons, on the login node (shared env python, PYTHONPATH = the staged tree).

  p9x_analyze.py G1_RESULTS G2_RUNS G2_RESULTS G3_RUNS G3_RESULTS > SUMMARY.txt
(G3 holds the one-GPU runs (a), the builds (b), the in-process two-rank runs d-* and the timing t-*; G2 the
subprocess two-rank runs c-*; G1 the first subprocess refusals.)

(a) embedding files with vs without --residue-codes, same GPU, bytes (a .json.gz but for its gzip mtime, bytes 4-7);
    the plain-repeat control; the folder listings.
(b) the codes of every command x format run (and of the timing runs on S2) against the residue-code store of
    'fm-search build sequences --residue-codes' on the same FASTA (ResidueCodeStore of the staged tree).
(c) two-rank runs: file names, per-rank ids (disjoint, union = S1), codes per id = the one-GPU run's, embeddings per
    id = the one-GPU run's (values; csv/pt files by bytes), the pass record of the code Parquet.
(d) refusal runs: exit code, time to exit, ESM3 markers, output folder unchanged, the message.
Timing: the t-* runs.
"""
import gzip
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

R1, G2, R2, G3, R3 = (Path(p) for p in sys.argv[1:6])
G1 = G3
FMTS = ("csv", "pt", "parquet", "json")
CMDS = ("residue", "chain")
CODE_RE = re.compile(r"\.codes\.(csv|pt)$")


def fasta_ids(path):
    ids, lens, cur = [], {}, None
    for line in open(path):
        if line.startswith(">"):
            cur = line[1:].split()[0]
            ids.append(cur)
            lens[cur] = 0
        else:
            lens[cur] += len(line.strip())
    return ids, lens


S1_IDS, S1_LEN = fasta_ids("/pscratch/sd/j/jsegura/embcomp/p9x/fx/s1.fasta")
S2_IDS, S2_LEN = fasta_ids("/pscratch/sd/j/jsegura/embcomp/p9x/fx/s2.fasta")


def same_bytes(a, b):
    x, y = Path(a).read_bytes(), Path(b).read_bytes()
    if str(a).endswith(".gz"):
        return x[:4] + x[8:] == y[:4] + y[8:]
    return x == y


def is_code_file(name, fmt):
    if fmt in ("csv", "pt"):
        return bool(CODE_RE.search(name))
    return name.startswith("inference-codes-")


def read_codes(folder, fmt, ids):
    """{id: uint8 [L, 8]} and {id: rank or None}."""
    out, rank = {}, {}
    if fmt == "parquet":
        for path in sorted(folder.glob("inference-codes-*.parquet")):
            r = int(path.stem.rsplit("-", 1)[1])
            t = pq.read_table(path)
            for i, res in zip(t.column("id").to_pylist(), t.column("residue_codes").to_pylist()):
                assert i not in out, f"{i} twice ({path})"
                out[i] = np.array(res, dtype=np.uint8).reshape(-1, 8)
                rank[i] = r
    elif fmt == "json":
        for path in sorted(folder.glob("inference-codes-*.json.gz")):
            r = int(path.name.split("-")[-1].split(".")[0])
            for rec in json.loads(gzip.decompress(path.read_bytes())):
                assert rec["id"] not in out, f"{rec['id']} twice ({path})"
                out[rec["id"]] = np.array(rec["residue_codes"], dtype=np.uint8).reshape(-1, 8)
                rank[rec["id"]] = r
    else:
        import torch
        for i in ids:
            p = folder / f"{i}.codes.{fmt}"
            if not p.exists():
                continue
            if fmt == "pt":
                t = torch.load(p, weights_only=True)
                assert t.dtype == torch.uint8 and t.device.type == "cpu" and t.dim() == 2 and t.shape[1] == 8, (p, t.dtype, t.shape)
                out[i] = t.numpy()
            else:
                text = p.read_text()
                out[i] = np.zeros((0, 8), np.uint8) if not text else pd.read_csv(p, header=None).to_numpy().astype(np.uint8)
            rank[i] = None
    return out, rank


def store_codes(db_dir, ids):
    from foldmatch.search.residue_code_store import ResidueCodeStore
    store = ResidueCodeStore(db_dir, "db")
    store.check_compatible()
    mm = np.asarray(store.open_memmap())
    entries = store.lookup(ids)
    return {i: np.array(mm[entries[i][0]:entries[i][0] + entries[i][1]]) for i in ids}


def compare_codes(got, ref, lens):
    ids = list(ref)
    missing = [i for i in ids if i not in got]
    extra = [i for i in got if i not in ref]
    neq = [i for i in ids if i in got and not np.array_equal(got[i], ref[i])]
    badlen = [i for i in ids if i in got and got[i].shape != (lens[i], 8)]
    bits = sum(int((np.unpackbits(got[i], axis=1) == np.unpackbits(ref[i], axis=1)).sum())
               for i in ids if i in got and got[i].shape == ref[i].shape)
    total = sum(ref[i].size * 8 for i in ids)
    return dict(n=len(got), missing=len(missing), extra=len(extra), differ=len(neq), bad_shape=len(badlen),
                bits=f"{bits}/{total}", equal=not (missing or extra or neq), first_diff=neq[:3])


def runs(results):
    recs = {}
    for line in open(results / "runs.jsonl"):
        r = json.loads(line)
        recs[r["label"]] = r
    return recs


def log_tail(results, label, n=6):
    lines = (results / "logs" / f"{label}.log").read_text().splitlines()
    msg = [ln[10:] for ln in lines if not ln.startswith("#")]
    return " | ".join(s.strip() for s in msg[-n:])


def p(*a):
    print(*a, flush=True)


def parquet_embeddings(folder):
    tabs = [pq.read_table(x) for x in sorted(folder.glob("inference-[0-9]*.parquet"))]
    out = {}
    for t in tabs:
        ids = t.column("id").to_pylist()
        col = t.column("embedding")
        for k, i in enumerate(ids):
            assert i not in out
            out[i] = col[k]
    return out


def json_chunks(folder):
    out = {}
    for path in sorted(folder.glob("inference-[0-9]*.json.gz")):
        text = gzip.decompress(path.read_bytes()).decode()
        starts = [m.start() for m in re.finditer(r'\{"id":', text)]
        for k, s in enumerate(starts):
            chunk = text[s:(starts[k + 1] if k + 1 < len(starts) else len(text))].rstrip("],")
            i = json.loads('"' + chunk[6:].split('"', 2)[1] + '"')
            assert i not in out
            out[i] = chunk
    return out




rec1, rec2, rec3 = runs(R1), runs(R2), runs(R3)
p("=" * 100)
p("RUNS")
for name, recs in (("G1", rec1), ("G2", rec2), ("G3", rec3)):
    bad = [(k, r["rc"]) for k, r in recs.items() if not k.startswith(("r", "c-r", "d-r", "c5")) and r["rc"] != 0]
    p(f"{name}: {len(recs)} records; non-refusal runs with rc != 0: {bad or 'none'}")

# ------------------------------------------------------------------------------------------------------------- (a)
p("=" * 100)
p("(a) embedding files, plain vs --residue-codes, same GPU, same process (in-process stream), S1 (101 sequences)")
for c in CMDS:
    for f in FMTS:
        plain, codes = G3 / f"a-{c}-{f}-plain", G3 / f"a-{c}-{f}-codes"
        if f"a-{c}-{f}-codes" not in rec3 or f"a-{c}-{f}-plain" not in rec3:
            p(f"  {c} {f}: not run")
            continue
        pn = sorted(x.name for x in plain.iterdir())
        cn = sorted(x.name for x in codes.iterdir())
        emb = [n for n in cn if not is_code_file(n, f)]
        code_files = [n for n in cn if is_code_file(n, f)]
        same = [n for n in pn if (codes / n).exists() and same_bytes(plain / n, codes / n)]
        gz = ""
        if f == "json":
            gz = f"; raw bytes equal incl. gzip mtime: {[(plain / n).read_bytes() == (codes / n).read_bytes() for n in pn]}"
        p(f"  {c:7s} {f:7s}: plain {len(pn)} files, codes run {len(emb)} embedding + {len(code_files)} code files; "
          f"embedding names equal: {pn == emb}; byte-identical: {len(same)}/{len(pn)}{gz}"
          + ("" if f in ("csv", "pt") else f"; code files {code_files}"))
for c in CMDS:
    a, b = G3 / f"a-{c}-parquet-plain", G3 / f"a-{c}-parquet-plain2"
    if b.exists():
        n = sorted(x.name for x in a.iterdir())
        p(f"  control {c} parquet plain vs plain repeat after the codes runs: "
          f"{sum(same_bytes(a / x, b / x) for x in n)}/{len(n)} byte-identical")
for c in CMDS:
    a, b = G3 / f"a-{c}-parquet-plain", G2 / f"c-{c}-parquet-codes"

# ------------------------------------------------------------------------------------------------------------- (b)
p("=" * 100)
p("(b) codes vs the residue-code store of 'fm-search build sequences --residue-codes' (same FASTA, A100)")
all_codes = {}
if (G3 / "b-build-s1" / "db" / "db.rescodes.json").exists() or "b-build-s1" in rec3:
    ref1 = store_codes(G3 / "b-build-s1" / "db", S1_IDS)
    for c in CMDS:
        for f in FMTS:
            if f"a-{c}-{f}-codes" in rec3:
                got, _ = read_codes(G3 / f"a-{c}-{f}-codes", f, S1_IDS)
                all_codes[c, f] = got
                p(f"  S1 {c:7s} {f:7s}: {compare_codes(got, ref1, S1_LEN)}")
    p(f"  empty [0, 8] tables in the S1 store: {sum(1 for v in ref1.values() if v.shape[0] == 0)}")
if "b-build-s2" in rec3 and rec3["b-build-s2"]["rc"] == 0:
    ref2 = store_codes(G3 / "b-build-s2" / "db", S2_IDS)
    for k in sorted(x for x in rec3 if x.startswith("t-") and "codes" in x):
        got, _ = read_codes(G3 / k, "parquet", S2_IDS)
        p(f"  S2 {k}: {compare_codes(got, ref2, S2_LEN)}")
    p(f"  empty [0, 8] tables in the S2 store: {sum(1 for v in ref2.values() if v.shape[0] == 0)}")
for c in CMDS:
    path = G3 / f"a-{c}-parquet-codes" / "inference-codes-0.parquet"
    if path.exists():
        t = pq.read_table(path)
        md = t.schema.metadata
        p(f"  {c} code Parquet schema: {t.schema.to_string(show_schema_metadata=False).replace(chr(10), '; ')}")
        p(f"    metadata keys: {sorted(k.decode() for k in md)}")
        p(f"    foldmatch.pass: {md.get(b'foldmatch.pass', b'').decode()}")
        import hashlib
        seqs = {}
        cur = None
        for line in open("/pscratch/sd/j/jsegura/embcomp/p9x/fx/s1.fasta"):
            if line.startswith(">"):
                cur = line[1:].split()[0]; seqs[cur] = ""
            else:
                seqs[cur] += line.strip()
        sha = dict(zip(t.column("id").to_pylist(), t.column("seq_sha1").to_pylist()))
        p(f"    seq_sha1 = sha1(sequence) for {sum(sha[i] == hashlib.sha1(seqs[i].encode()).digest() for i in sha)}/{len(sha)} ids")
path = G3 / "a-chain-parquet-codes" / "inference-0.parquet"
if path.exists():
    emb_t = pq.read_schema(path)
    p(f"  chain embedding Parquet of a codes run: {emb_t.to_string(show_schema_metadata=False).replace(chr(10), '; ')}; metadata {emb_t.metadata}")

# ------------------------------------------------------------------------------------------------------------- (c)
p("=" * 100)
p("(c) two ranks (srun -n 2, DDP) vs one GPU, --residue-codes, S1")


def ddp_check(folder, rc, c, f):
    d1 = G3 / f"a-{c}-{f}-codes"
    names = sorted(x.name for x in folder.iterdir())
    got, rank = read_codes(folder, f, S1_IDS)
    cmp = compare_codes(got, all_codes[c, f], S1_LEN) if (c, f) in all_codes else "no 1-GPU run"
    line = f"  {c:7s} {f:7s}: rc {rc}; {len(names)} files; codes vs 1 GPU: {cmp}"
    if f in ("parquet", "json"):
        per = {r: sum(1 for v in rank.values() if v == r) for r in sorted(set(rank.values()))}
        line += f"; files {names}; ids per rank {per}"
        if f == "parquet":
            e2, e1 = parquet_embeddings(folder), parquet_embeddings(d1)
            eq = sum(1 for i in e1 if i in e2 and e1[i].equals(e2[i]))
            line += f"; embeddings per id equal to 1 GPU: {eq}/{len(e1)} (2-rank ids {len(e2)})"
            for r in (0, 1):
                fp = folder / f"inference-codes-{r}.parquet"
                if fp.exists():
                    rec = json.loads(pq.read_schema(fp).metadata[b"foldmatch.pass"])
                    line += f"; rank {r} pass world_size {rec.get('world_size')}"
        else:
            e2, e1 = json_chunks(folder), json_chunks(d1)
            eq = sum(1 for i in e1 if i in e2 and e1[i] == e2[i])
            line += f"; embedding records per id equal to 1 GPU (text): {eq}/{len(e1)} (2-rank ids {len(e2)})"
    else:
        emb = [n for n in names if not is_code_file(n, f)]
        eq = sum(1 for n in emb if (d1 / n).exists() and same_bytes(d1 / n, folder / n))
        line += f"; {len(emb)} embedding + {len(names) - len(emb)} code files; embedding files byte-equal to 1 GPU: {eq}/{len(emb)}"
    p(line)


p(" subprocess runs (job G2, one CLI process per rank):")
for c in CMDS:
    for f in FMTS:
        k = f"c-{c}-{f}-codes"
        if k in rec2:
            try:
                ddp_check(G2 / k, rec2[k]["rc"], c, f)
            except Exception as e:  # noqa: BLE001
                p(f"  {c} {f}: check failed: {e!r}")
p(" in-process runs (job G3, one process per rank running all eight):")
for c in CMDS:
    for f in FMTS:
        k = f"d-{c}-{f}-codes"
        rcs = [rec3[f"{k}@r{r}"]["rc"] for r in (0, 1) if f"{k}@r{r}" in rec3]
        if rcs:
            try:
                ddp_check(G3 / k, rcs, c, f)
            except Exception as e:  # noqa: BLE001
                p(f"  {c} {f}: check failed: {e!r}")
for k, r in sorted({**{x: (rec2[x], R2) for x in rec2 if x.startswith("c-r")},
                    **{x: (rec3[x], R3) for x in rec3 if x.startswith("d-r")}}.items()):
    r, R = r
    p(f"  refusal under 2 ranks {k}: rc {r['rc']}, {r['wall']} s, ESM3 reached {r['esm3_reached']}, "
      f"out unchanged {r['out_unchanged']}; {(r.get('message') or log_tail(R, k, 4))[:300]}")

# ------------------------------------------------------------------------------------------------------------- (d)
p("=" * 100)
p("(d) refusals: exit code, time to exit, ESM3 reached ('Loading rcsb-esm' logged), output folder unchanged")
for name, recs, R in (("G1 subprocess", rec1, R1), ("G3", rec3, R3)):
    for k, r in recs.items():
        if k.startswith(("r", "c5")):
            msg = r.get("message") or log_tail(R, k, 3)
            p(f"  {name:13s} {k:12s} rc {r['rc']} {r['wall']:8.3f} s  ESM3 reached {r['esm3_reached']!s:5s} "
              f"out unchanged {r['out_unchanged']!s:5s} {'in-process' if r.get('inproc') else 'subprocess'} | {msg[:330]}")
le = [(k, r["marks"]["loading_esm"], r["marks"].get("esm_ready")) for k, r in rec3.items()
      if k.startswith(("a-", "t-")) and "loading_esm" in r["marks"]]
if le:
    xs = sorted(x[1] for x in le)
    p(f"  in-process normal runs: 'Loading rcsb-esm' at {xs[0]:.2f}-{xs[-1]:.2f} s after the command starts (n={len(xs)})")
for name, recs in (("G2", rec2),):
    for k, r in recs.items():
        if "loading_esm" in r["marks"]:
            p(f"  {name} subprocess {k}: 'Loading rcsb-esm' at {r['marks']['loading_esm']:.1f} s, ready at {r['marks'].get('esm_ready')}")

# ------------------------------------------------------------------------------------------------------------ timing
p("=" * 100)
p("Timing, S2 (301 sequences, 113,385 residues), parquet, GPU 0, alone on the node (phase 2), in-process "
  "(imports excluded); wall = command start to return; esm load = 'Loading rcsb-esm' to 'module ready'; "
  "after = 'module ready' to return (pass + writing)")
for k in sorted(x for x in rec3 if x.startswith("t-") or x.startswith("b-build-s2")):
    r = rec3[k]
    m = r["marks"]
    load = m.get("esm_ready", 0) - m.get("loading_esm", 0)
    after = r["wall"] - m["esm_ready"] if "esm_ready" in m else None
    p(f"  {k:20s} rc {r['rc']} wall {r['wall']:7.2f} s  esm load {load:6.2f} s  after {after if after is None else round(after, 2)} s"
      f"  CUDA peak {r.get('cuda_peak_alloc_gib')} GiB  import {r.get('stream_import_s')} s  started {r['started']}")
p("One-GPU (a) runs, phase 1 (streams A, B in parallel):")
for k in sorted(x for x in rec3 if x.startswith(("a-", "b-build-s1"))):
    r = rec3[k]
    m = r["marks"]
    after = r["wall"] - m["esm_ready"] if "esm_ready" in m else None
    p(f"  {k:28s} rc {r['rc']} gpu {r['gpu']} wall {r['wall']:7.2f} s after-ready {after if after is None else round(after, 2)} s "
      f"CUDA peak {r.get('cuda_peak_alloc_gib')} GiB")
