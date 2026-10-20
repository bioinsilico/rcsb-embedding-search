#!/usr/bin/env python
"""Self-hit study, step 2: FoldMatch's own ESM3 + esm-codes-v1 head for a FASTA, giving each sequence BOTH its query
encoding (q fp16, S5) and its target code (uint8, S4), so its self hit can be scored without a database.

FoldMatch is used read-only from a tree on PYTHONPATH (the P6 sandbox tree ``wt-P6/src``): ``esm_open`` +
``per_residue_embedding`` (geometric attention off for sequence-only input, as the build and query passes) and the
registered ``esm-codes-v1`` head (``scoring_system.prepare(...).load_head()``), one protein at a time, rows 1..L.

    PYTHONPATH=<wt-P6/src> HF_HUB_OFFLINE=1 HF_HUB_CACHE=<cache> \
        python encode_fm.py --fasta F [--fasta F2 ...] --out OUT.npz [--device mps|cpu] [--flush 50]

Output npz: ids, headers (UTF-8 concat + offsets), lengths, offsets, q fp16 [N_res, 64], codes uint8 [N_res, 8],
seq_concat uint8, seconds per sequence, meta_json. Partial results go to OUT.part.npz every --flush sequences and a
rerun resumes from it (heavy runs on the Mac crash-recover this way).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np


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


def pack(records, qs, cs, secs, meta):
    ids = np.array([h.split()[0] for h, _ in records])
    heads = [h.encode("utf-8") for h, _ in records]
    hoff = np.zeros(len(heads) + 1, np.int64)
    np.cumsum([len(h) for h in heads], out=hoff[1:])
    lens = np.array([len(s) for _, s in records], np.int64)
    off = np.zeros(len(lens) + 1, np.int64)
    np.cumsum(lens, out=off[1:])
    return dict(ids=ids, header_concat=np.frombuffer(b"".join(heads), np.uint8), header_offsets=hoff,
                lengths=lens, offsets=off,
                q=np.concatenate(qs) if qs else np.zeros((0, 64), np.float16),
                codes=np.concatenate(cs) if cs else np.zeros((0, 8), np.uint8),
                seq_concat=np.frombuffer("".join(s for _, s in records).encode("ascii"), np.uint8),
                seconds=np.array(secs, np.float64), meta_json=json.dumps(meta))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--flush", type=int, default=50)
    args = ap.parse_args()

    import torch
    from esm.sdk.api import ESMProtein
    from foldmatch.search import scoring_system as ss
    from foldmatch.utils.esm.loaders import esm_open, per_residue_embedding
    from foldmatch.utils.model import release_mps_cache
    from foldmatch.utils.sequence import normalize_sequence

    records = []
    for f in args.fasta:
        records += [(h, normalize_sequence(s)) for h, s in read_fasta(f)]
    seen = set()
    for h, _ in records:
        key = h.split()[0]
        if key in seen:
            sys.exit(f"duplicate id {key}")
        seen.add(key)

    part = args.out + ".part.npz"
    qs, cs, secs, done = [], [], [], 0
    if os.path.exists(part):
        z = np.load(part, allow_pickle=False)
        done = len(z["lengths"])
        if [str(x) for x in z["ids"]] != [h.split()[0] for h, _ in records[:done]]:
            sys.exit(f"{part} is not a prefix of this input")
        off = z["offsets"]
        qs = [z["q"][off[i]:off[i + 1]] for i in range(done)]
        cs = [z["codes"][off[i]:off[i + 1]] for i in range(done)]
        secs = list(z["seconds"])
        print(f"resuming after {done} sequences", flush=True)

    device = torch.device(args.device)
    system = ss.get("esm-codes-v1")
    prepared = ss.prepare(system, local_files_only=True)
    t0 = time.time()
    model = esm_open(device)
    head = prepared.load_head(device=device)
    print(f"models on {device} in {time.time() - t0:.1f}s; {len(records) - done} sequences to encode", flush=True)
    meta = dict(system=system.name, system_digest=system.digest(), device=str(device), torch=torch.__version__,
                fasta=args.fasta, foldmatch_path=os.path.dirname(ss.__file__))

    t_start = time.time()
    with torch.inference_mode():
        for i in range(done, len(records)):
            h, s = records[i]
            t1 = time.time()
            emb = per_residue_embedding(model, ESMProtein(sequence=s))
            if emb.shape[0] != len(s) + 2:
                sys.exit(f"{h.split()[0]}: ESM3 gave {tuple(emb.shape)} for {len(s)} residues")
            rows = emb[1:-1]
            qs.append(head.encode_query(rows).cpu().numpy().astype(np.float16, copy=False))
            cs.append(head.encode_target(rows).cpu().numpy().astype(np.uint8, copy=False))
            release_mps_cache(device)
            secs.append(time.time() - t1)
            if (i + 1) % args.flush == 0 or i + 1 == len(records):
                np.savez(part, **pack(records[:i + 1], qs, cs, secs, meta))
                res = sum(len(x) for _, x in records[done:i + 1])
                print(f"{i + 1}/{len(records)} done, {res / (time.time() - t_start):.0f} residues/s", flush=True)
    np.savez(args.out, **pack(records, qs, cs, secs, meta))
    os.remove(part)
    print(f"wrote {args.out}: {len(records)} sequences, {sum(len(x) for _, x in records)} residues", flush=True)


if __name__ == "__main__":
    main()
