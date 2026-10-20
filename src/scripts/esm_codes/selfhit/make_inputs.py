#!/usr/bin/env python
"""Self-hit study, step 1: the FASTA inputs of the local encoding pass (encode_fm.py). numpy only.

* ``u50_q200.fasta``: the 200 P6 check queries (100 in the 10k UniRef50 database, 100 held out), unchanged.
* ``u50_db800.fasta``: 800 entries drawn at random (seed 20261019) from the 10k database (itself a uniform random
  draw over UniRef50 2026_03 clusters of <= 2000 aa), excluding the 100 in-database queries: the natural UniRef50
  population, not length-stratified.
* ``scop_q.fasta``: the unique sequences of the 1,465 SCOP benchmark queries the E-values were calibrated on
  (``evalue/encode/out/queries.L2m_B8_s0.npz``), first record of each sequence.
* ``pdb_r600.fasta`` (``--pdb`` only): 600 of the 2,000 uniform random unique PDB seqres protein sequences of the P0.6
  census (``p0/census/seq/items.tsv``, pdb category ``random``), seed 20261019.

    python make_inputs.py DATA_DIR      (DATA_DIR holds p6/queries200.fasta, u10k/, scop/; writes DATA_DIR/inputs/)
    python make_inputs.py DATA_DIR --pdb   (DATA_DIR/census/items.tsv; writes only pdb_r600.fasta)
"""
from __future__ import annotations

import json
import os
import random
import sys

import numpy as np

SEED = 20261019
N_DB = 800


def read_fasta(path):
    out, name, buf = [], None, []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if name is not None:
                    out.append((name, "".join(buf)))
                name, buf = line[1:], []
            elif name is not None:
                buf.append(line.strip())
    if name is not None:
        out.append((name, "".join(buf)))
    return out


def write_fasta(path, records):
    with open(path, "w", encoding="utf-8") as fh:
        for h, s in records:
            fh.write(f">{h}\n")
            for i in range(0, len(s), 80):
                fh.write(s[i:i + 80] + "\n")


def pdb_sample(d, out, n=600):
    rows = []
    with open(os.path.join(d, "census", "items.tsv"), encoding="utf-8") as fh:
        cols = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            r = dict(zip(cols, line.rstrip("\n").split("\t")))
            if "random" in r["pdb_cats"].split(","):
                rows.append(r)
    pick = sorted(random.Random(SEED).sample(range(len(rows)), n))
    recs = [(f"pdb_{rows[i]['item']} {rows[i]['pdb_ids'].split(',')[0]} cats={rows[i]['pdb_cats']}",
             rows[i]["sequence"]) for i in pick]
    write_fasta(os.path.join(out, "pdb_r600.fasta"), recs)
    print(json.dumps({"pdb_r600": {"records": len(recs), "residues": sum(len(s) for _, s in recs),
                                   "pool": len(rows)}}))


def main():
    d = sys.argv[1]
    out = os.path.join(d, "inputs")
    os.makedirs(out, exist_ok=True)
    if sys.argv[2:] == ["--pdb"]:
        pdb_sample(d, out)
        return
    q200 = read_fasta(os.path.join(d, "p6", "queries200.fasta"))
    write_fasta(os.path.join(out, "u50_q200.fasta"), q200)

    db = read_fasta(os.path.join(d, "u10k", "uniref50_10k.fasta"))
    qids = {h.split()[0] for h, _ in q200}
    pool = [r for r in db if r[0].split()[0] not in qids]
    pick = sorted(random.Random(SEED).sample(range(len(pool)), N_DB))
    db800 = [pool[i] for i in pick]
    write_fasta(os.path.join(out, "u50_db800.fasta"), db800)

    z = np.load(os.path.join(d, "scop", "queries.L2m_B8_s0.npz"), allow_pickle=False)
    off, sc = z["offsets"], z["seq_concat"].tobytes().decode("ascii")
    hc, ho = z["header_concat"].tobytes().decode("utf-8"), z["header_offsets"]
    seen, scop = set(), []
    for i in range(len(off) - 1):
        s = sc[off[i]:off[i + 1]]
        if s in seen:
            continue
        seen.add(s)
        scop.append(("scop_" + hc[ho[i]:ho[i + 1]].strip(), s))
    write_fasta(os.path.join(out, "scop_q.fasta"), scop)

    summary = {name: {"records": len(r), "residues": sum(len(s) for _, s in r)}
               for name, r in (("u50_q200", q200), ("u50_db800", db800), ("scop_q", scop))}
    summary.update(seed=SEED, db_pool=len(pool))
    with open(os.path.join(out, "inputs.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
