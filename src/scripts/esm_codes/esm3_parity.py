#!/usr/bin/env python
"""P0.5 ESM3 parity (fixture F6) and the A5 re-embedding for FoldMatch esm-codes-v1 (plan FINAL v1, section 4 P0.5).

Question: do residue codes / query encodings computed through FoldMatch's OWN ESM3 residue path (fp32 weights loaded by
foldmatch.utils.esm.loaders.esm_open from rcsb/rcsb-esm, one protein at a time through
ChainCompleteModule.predict_step -> ESM3.forward_and_sample(..., SamplingConfig(return_per_residue_embeddings=True)),
bf16 autocast only inside ESM3.logits on CUDA, plain fp32 on CPU / MPS) and then the research zcw50_clean + M8_s0
(L2m_B8_s0) encoders reproduce the research encodings (bf16-weight batched SequenceOnlyESM3 on A100) closely enough for
the shipped E-value table, and how do devices mix?

Pipeline per protein (normative spec S1-S5, FoldMatch path):
  FASTA (foldmatch.dataset.esm_prot_from_fasta.EsmProtFromFasta, FoldMatch's own reader) -> ChainCompleteModule
  .predict_step([(ESMProtein, name)]) with a capture module in the aggregator slot (it receives exactly what the
  aggregator receives: collate_seq_embeddings of per_residue_embedding [L+2, 1536]) -> rows 1..L -> fp16 ->
  zcw50_clean (the sw_common.Whitener arithmetic: fp32 math, fp16 out) -> f_q -> fp16 (q) / f_t -> sign ->
  np.packbits (codes). The encoders (sw_common.build_model, L2m_B8_s0.pt) run on the ESM3 device in fp32, TF32 off,
  outside any autocast (spec S3), one protein per call. No FoldMatch file is modified; FoldMatch is imported from
  FM_SRC (Mac: the checkout; NERSC: a copy of the same src/ tree, see FM_COMMIT).

Subcommands (all paths resolve on the Mac and on NERSC; see Paths):
  inputs                 write inputs/*.fasta (unique sequences, header = sha1) + inputs/sets.json (record order, md5s)
                         sets: panel (1,000 shipped members), enccheck (50 enc_check items), q200 (200 benchmark
                         queries stratified by length), chains (559 chain-set chains), tsub (targets of the sampled
                         research E<=1 hits of q200: <= HITS_PER_QUERY per query), a5q (1,465 queries), a5t
                         (50,000 targets)
  embed --device D --sets S.. [--rank r --nranks n] [--raw S..]
                         FoldMatch ESM3 + research encoders -> emb/<devtag>/<set>/r<rank>_c<k>.npz (resumable,
                         DEADLINE env = epoch seconds); --raw also keeps the fp16 rows 1..L for those sets;
                         --mps-empty-cache releases the MPS allocator cache after every protein (needed on a 64 GB Mac)
  merge --devtag T --sets S..   parts -> enc/<devtag>.<set>.npz in the set's RECORD order (q fp16, codes, offsets...)
  score --plan NAME      NERSC CPU: Smith-Waterman setting D, S_f and S_r (research benchmark-driver arithmetic, research
                         kernel swkern.c/swmulti.c) for the planned (query source x target source x pair set) jobs
  chainf1 --srcs Q[@T].. chain_eval.evaluate (732 chain pairs, harness DP, setting D) for encoding sources
  analyze                all metrics -> esm3_parity.json
  thresholds             esm3_parity.json -> A4_THRESHOLDS.json (next to this script; gate values + reasons are the
                         A4_* constants and the text in cmd_thresholds)
  archive --dest DIR     compact durable copy of the outputs (default NERSC home esm_codes_archive/p0/parity) + MD5SUMS

Encoding sources ("src"): research (the research encodings), gpu (NERSC A100, bf16 autocast), cpu_linux (NERSC EPYC
7763 CPU fp32), cpu_mac (M1 Max CPU fp32), mps (M1 Max MPS fp32).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import platform
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")          # never download: the rcsb/rcsb-esm snapshot must be cached
# BLAS threads: set OMP_NUM_THREADS / OPENBLAS_NUM_THREADS=1 in the environment for score / chainf1 (fork pools);
# embed sets torch threads explicitly (--threads).

import numpy as np  # noqa: E402

# =============================================================================
# Paths
# =============================================================================
SP_LOCAL = "/private/tmp/claude-501/-Users-joan-devel-rcsb-embedding-search/d9266e69-77ea-4121-a428-2e6f3bcc4ca2/scratchpad"
IS_NERSC = os.path.isdir("/pscratch/sd/j/jsegura")
EMB = os.environ.get("EMBCOMP_DATA") or ("/pscratch/sd/j/jsegura/embcomp" if IS_NERSC else f"{SP_LOCAL}/embcomp")
WORK = os.environ.get("PARITY_WORK") or (f"{EMB}/p0/parity" if IS_NERSC else f"{SP_LOCAL}/p0/parity")
FM_SRC = os.environ.get("FM_SRC") or (f"{WORK}/fm_src" if IS_NERSC else "/Users/joan/devel/rcsb-embedding-model/src")
KERNEL = os.environ.get("PARITY_KERNEL") or (f"{WORK}/kernel" if IS_NERSC else f"{EMB}/panel/fit/kernel")
SWM = f"{EMB}/sw_mlp"
MODEL = f"{SWM}/models/L2m_B8_s0.pt"
ZCW = f"{EMB}/clean_pairs/zcw50_clean.npz"
REF = {
    "queries": f"{EMB}/evalue/encode/out/queries.L2m_B8_s0.npz",
    "targets": (f"{EMB}/evalue/encode/out/targets.L2m_B8_s0.npz" if IS_NERSC
                else f"{EMB}/evalue/verify/data/targets.L2m_B8_s0.npz"),
    "panel": f"{EMB}/panel/fit2/results/shipped_panel_codes.npz",
    "chains": (f"{SWM}/enc/L2m_B8_s0.chains.npz" if IS_NERSC else f"{EMB}/throughput/bench/data/L2m_B8_s0.chains.npz"),
    "enccheck": f"{SWM}/enc_check/L2m_B8_s0.npz",
    "enccheck_items": f"{SWM}/enc_check/check_items.json",
    "chain_emb": f"{EMB}/chains/data/chain_emb.npz",
    "val": f"{EMB}/val.npz",
    "test": f"{EMB}/test.npz",
}
SHIP = f"{EMB}/panel/fit2/results/ship_final.json"
SHIP_KEY = "D_fmr/pnl_uniref50/M3t999"
BENCH_SCORE = f"{EMB}/evalue/score/out"                  # research S_D_fwd / S_D_rev [1465, 50000], index.npz
LABELS = f"{EMB}/evalue/score/results/labels.npz"
Q90_SHIPPED = f"{EMB}/panel/fit2/results/bench_q90_shipped.npy"
CHAIN_REF_D = f"{EMB}/evalue/f1_settings/results/chains/M8_s0_D.npz"
SETTING_D = (1.0, 5.0, 0.5)                             # offset, gap open, gap extend
EDGES = [0, 40, 60, 100, 200, 400, 10 ** 9]              # M3 query-length bins (rlib.EDGES)
Q200_ALLOC = [34, 34, 33, 33, 33, 33]
Q200_SEED = 20260927
HITS_PER_QUERY = 8                                       # tsub: <= 8 research E<=1 hits per q200 query
HITS_SEED = 20260928
STD = set("ACDEFGHIKLMNPQRSTVWY")
SRCS = ("research", "gpu", "cpu_linux", "cpu_mac", "mps")
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:8.1f}s] " + " ".join(str(x) for x in a), flush=True)


def md5(path, bs=1 << 24):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(bs), b""):
            h.update(b)
    return h.hexdigest()


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("ascii")).hexdigest()


def normalize_sequence(s: str) -> str:
    """Spec S15 (ascii_upper_nows_v1): remove ASCII whitespace, uppercase ASCII letters; non-ASCII refused."""
    s.encode("ascii")
    return "".join(s.split()).upper()


def jdump(obj, path):
    def conv(o):
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, default=conv)
    os.replace(tmp, path)


def read_fasta(path):
    recs, h, s = [], None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if h is not None:
                    recs.append((h, "".join(s)))
                h, s = line[1:], []
            elif line.strip():
                s.append(line.strip())
    if h is not None:
        recs.append((h, "".join(s)))
    return recs


def load_concat_set(path):
    """research encode.py format -> ids, sequences (lists), offsets, npz handle."""
    z = np.load(path, mmap_mode="r")
    off = np.asarray(z["offsets"])
    sc = np.asarray(z["seq_concat"]).tobytes().decode("ascii")
    ids = z["ids"].astype(str).tolist()
    seqs = [sc[off[i]:off[i + 1]] for i in range(len(ids))]
    return ids, seqs, off, z


# =============================================================================
# research E-value procedure (evalue/reconcile/ship.py, panel/fit2 bench_validate.py)
# =============================================================================
def ship_table():
    t = json.load(open(SHIP))["tables"][SHIP_KEY]
    return t


def tab_params(tab, m):
    lm = np.log(np.asarray(m, np.float64))
    lnc = np.log(np.asarray(tab["center_len"], np.float64))
    return np.interp(lm, lnc, tab["lam"]), np.interp(lm, lnc, tab["lnK"])


def ln_evalue(tab, m, S, q90, N):
    """lnE = lnK(m) + ln m + ln N - lambda(m) (S - Q90), rows = queries (m [nq]), S [nq, nt] (float64)."""
    lam, lnK = tab_params(tab, m)
    m = np.asarray(m, np.float64)
    return (lnK + np.log(m) + math.log(N))[:, None] - lam[:, None] * (S - np.asarray(q90)[:, None])


class BenchLabels:
    """calib3di benchmark labels (the rlib.Bench fields used here), loaded without rlib's local-only paths."""

    def __init__(self):
        ix = np.load(f"{BENCH_SCORE}/index.npz")
        lab = np.load(LABELS)
        self.qids = ix["query_ids"].astype(str)
        self.tids = ix["target_ids"].astype(str)
        assert np.array_equal(self.qids, lab["query_ids"].astype(str))
        assert np.array_equal(self.tids, lab["target_ids"].astype(str))
        self.nq, self.nt = (int(x) for x in lab["shape"])
        self.qlen = ix["query_len"].astype(np.float64)
        self.tlen = ix["target_len"].astype(np.float64)
        self.rev = lab["rev"].astype(bool)
        un = lambda k: np.unpackbits(lab[k], axis=1, count=self.nt).astype(bool)  # noqa: E731
        self.TP, self.REAL = un("TP"), un("FPA")
        assert int(self.TP.sum()) == 31099 and int(self.REAL.sum()) == 34948695
        self.qbin = np.digitize(self.qlen, EDGES[1:-1], right=False)
        self.N_ann = float(self.tlen[~self.rev].sum())
        self.N_all = float(self.tlen.sum())


def research_S(rows=None):
    f = np.load(f"{BENCH_SCORE}/S_D_fwd.npy", mmap_mode="r")
    r = np.load(f"{BENCH_SCORE}/S_D_rev.npy", mmap_mode="r")
    if rows is None:
        return np.asarray(f, np.float64) - np.asarray(r, np.float64)
    return np.asarray(f[rows], np.float64) - np.asarray(r[rows], np.float64)


# =============================================================================
# inputs
# =============================================================================
def select_q200(qlen):
    b = np.digitize(qlen, EDGES[1:-1], right=False)
    rng = np.random.default_rng(Q200_SEED)
    sel = [rng.choice(np.flatnonzero(b == k), Q200_ALLOC[k], replace=False) for k in range(6)]
    return np.sort(np.concatenate(sel))


def research_hits_q200(q200):
    """Research E<=1 hits of the 200 queries against all 50,000 benchmark targets (shipped table, shipped-panel Q90,
    N = annotated residues of the 50,000 columns, as bench_validate 'own' NEW), and the tsub sample."""
    BL = BenchLabels()
    tab = ship_table()["all"]
    q90 = np.load(Q90_SHIPPED)
    S = research_S(q200)
    L = ln_evalue(tab, BL.qlen[q200], S, q90[q200], BL.N_ann)
    rng = np.random.default_rng(HITS_SEED)
    pairs, allp = [], []
    for r, qi in enumerate(q200):
        h = np.flatnonzero(L[r] <= 0.0)
        allp += [(int(qi), int(t)) for t in h]
        if len(h) > HITS_PER_QUERY:
            h = np.sort(rng.choice(h, HITS_PER_QUERY, replace=False))
        pairs += [(int(qi), int(t)) for t in h]
    return pairs, allp


def cmd_inputs(_a):
    out = f"{WORK}/inputs"
    os.makedirs(out, exist_ok=True)
    sets = {}
    # panel
    zp = np.load(REF["panel"])
    p_ids, p_seq = zp["ids"].astype(str).tolist(), zp["sequences"].astype(str).tolist()
    fa = read_fasta(f"{EMB}/panel/fit2/results/shipped_panel.fasta")
    assert [h.split()[0] for h, _ in fa] == p_ids and [s for _, s in fa] == p_seq
    sets["panel"] = dict(ids=p_ids, seqs=p_seq, source=REF["panel"])
    # enc_check (15 val + 15 test domains + 20 chains)
    items = json.load(open(REF["enccheck_items"]))
    ids, seqs = [], []
    for split in ("val", "test"):
        z = np.load(REF[split])
        dids = z["domain_ids"].astype(str)
        pos = {d: k for k, d in enumerate(dids)}
        sq = z["sequences"]
        for d in items[split]:
            ids.append(f"{split}/{d}")
            seqs.append(str(sq[pos[d]]))
    zc = np.load(REF["chain_emb"])
    cpos = {c: k for k, c in enumerate(zc["chain_ids"].astype(str))}
    csq = zc["sequences"].astype(str)
    for c in items["chains"]:
        ids.append(f"chains/{c}")
        seqs.append(str(csq[cpos[c]]))
    ze = np.load(REF["enccheck"])
    for i, s in zip(ids, seqs):
        grp, name = i.split("/")
        assert ze[f"{grp}/code/{name}"].shape[0] == len(s), i
    sets["enccheck"] = dict(ids=ids, seqs=seqs, source=REF["enccheck"])
    # chains (L2m_B8_s0.chains.npz order)
    zr = np.load(REF["chains"])
    ch_ids = zr["ids"].astype(str).tolist()
    assert ch_ids == zc["chain_ids"].astype(str).tolist()
    ch_seq = [str(csq[cpos[c]]) for c in ch_ids]
    assert np.array_equal(np.diff(zr["offsets"]), [len(s) for s in ch_seq])
    sets["chains"] = dict(ids=ch_ids, seqs=ch_seq, source=REF["chains"])
    # benchmark queries / targets
    qids, qseq, qoff, _ = load_concat_set(REF["queries"])
    fq = read_fasta(f"{SP_LOCAL}/calib3di/results/queries.fasta" if not IS_NERSC
                    else "/pscratch/sd/j/jsegura/fm-db/calib3di/data/queries.fasta")
    assert [h.split()[0] for h, _ in fq] == qids and [s for _, s in fq] == qseq
    tids, tseq, toff, _ = load_concat_set(REF["targets"])
    sets["a5q"] = dict(ids=qids, seqs=qseq, source=REF["queries"])
    sets["a5t"] = dict(ids=tids, seqs=tseq, source=REF["targets"])
    qlen = np.diff(qoff)
    q200 = select_q200(qlen)
    sets["q200"] = dict(ids=[qids[i] for i in q200], seqs=[qseq[i] for i in q200], source=REF["queries"],
                        rows=q200.tolist())
    pairs, allp = research_hits_q200(q200)
    tsub = sorted({t for _, t in pairs})
    sets["tsub"] = dict(ids=[tids[t] for t in tsub], seqs=[tseq[t] for t in tsub], source=REF["targets"],
                        cols=tsub)
    manifest = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "host": platform.node(), "sets": {},
                "q200": dict(seed=Q200_SEED, alloc=Q200_ALLOC, edges=EDGES[:-1] + ["inf"], rows=q200.tolist()),
                "hits": dict(definition="research lnE <= 0 (E <= 1): shipped table all, Q90 = bench_q90_shipped.npy "
                                        "(shipped panel), N = annotated residues of the 50,000 benchmark targets; "
                                        f"<= {HITS_PER_QUERY} hits per query sampled with default_rng({HITS_SEED})",
                             n_research_hits_q200=len(allp), all_pairs=allp, pairs=pairs, n_pairs=len(pairs),
                             n_targets=len(tsub)),
                "sources_md5": {k: md5(v) for k, v in REF.items() if os.path.isfile(v) and not v.endswith(("val.npz", "test.npz"))}}
    for name, s in sets.items():
        norm = [normalize_sequence(x) for x in s["seqs"]]
        n_changed = sum(a != b for a, b in zip(norm, s["seqs"]))
        keys = [sha1(x) for x in norm]
        uniq = {}
        for k, x in zip(keys, norm):
            uniq.setdefault(k, x)
        path = f"{out}/{name}.fasta"
        with open(path, "w") as fh:
            for k, x in uniq.items():
                fh.write(f">{k}\n")
                for i in range(0, len(x), 80):
                    fh.write(x[i:i + 80] + "\n")
        nonstd = sum(bool(set(x) - STD) for x in norm)
        manifest["sets"][name] = dict(n_records=len(keys), n_unique=len(uniq), residues=int(sum(map(len, norm))),
                                      unique_residues=int(sum(map(len, uniq.values()))), fasta=path, fasta_md5=md5(path),
                                      normalisation_changed=n_changed, n_nonstandard_records=nonstd,
                                      source=s["source"], ids=s["ids"], keys=keys,
                                      **({"rows": s["rows"]} if "rows" in s else {}),
                                      **({"cols": s["cols"]} if "cols" in s else {}))
        log(f"{name}: {len(keys)} records, {len(uniq)} unique, {manifest['sets'][name]['residues']:,} residues, "
            f"normalisation changed {n_changed}, non-standard {nonstd}")
    jdump(manifest, f"{out}/sets.json")


def load_sets():
    return json.load(open(f"{WORK}/inputs/sets.json"))


# =============================================================================
# embed: FoldMatch ESM3 path + research encoders
# =============================================================================
def import_foldmatch():
    if FM_SRC not in sys.path:
        sys.path.insert(0, FM_SRC)
    import foldmatch
    assert os.path.realpath(foldmatch.__file__).startswith(os.path.realpath(FM_SRC)), foldmatch.__file__
    return foldmatch


def fm_provenance():
    files = ["foldmatch/utils/esm/loaders.py", "foldmatch/modules/chain_complete_module.py", "foldmatch/utils/model.py",
             "foldmatch/dataset/esm_prot_from_fasta.py", "foldmatch/utils/fasta.py", "foldmatch/utils/data.py",
             "foldmatch/inference/chain_complete_inference.py"]
    out = {"fm_src": FM_SRC, "files_md5": {f: md5(os.path.join(FM_SRC, f)) for f in files}}
    commit = os.path.join(FM_SRC, "..", "FM_COMMIT")
    if os.path.exists(commit):
        out["commit"] = open(commit).read().strip()
    return out


def esm_snapshot():
    from foldmatch.utils.esm.loaders import data_root
    root = data_root()
    w = os.path.join(root, "data/weights/esm3_sm_open_v1.pth")
    return {"snapshot": str(root), "weights_blob": os.path.basename(os.path.realpath(w))}


def build_capture():
    """nn.Module that stands in the aggregator slot of ChainCompleteModule and applies the research encoders."""
    import torch
    if SWM not in sys.path:
        sys.path.insert(0, SWM)
    os.environ.setdefault("EMBCOMP_DATA", EMB)
    import sw_common as C

    class CodeCapture(torch.nn.Module):
        def __init__(self):
            super().__init__()
            st = torch.load(MODEL, map_location="cpu")
            self.cfg = st["cfg"]
            self.enc = C.build_model(self.cfg)
            self.enc.load_state_dict(st["state"])
            self.enc.eval()
            keep, off, M = C.zcw50_clean_map()                       # float64 -> float32 once (spec S2)
            self.register_buffer("keep", torch.as_tensor(keep))
            self.register_buffer("off", torch.as_tensor(off, dtype=torch.float32))
            self.register_buffer("M", torch.as_tensor(M, dtype=torch.float32))
            self.out = []
            self.keep_raw = False

        @torch.no_grad()
        def forward(self, res_batch_embedding, res_batch_mask):
            # mask True = padding (foldmatch.utils.data.collate_seq_embeddings)
            for b in range(res_batch_embedding.shape[0]):
                e = res_batch_embedding[b][~res_batch_mask[b]]          # [L+2, 1536] float32
                rows = e[1:-1]                                         # BOS / EOS removed (spec S1)
                assert rows.dtype == torch.float32
                x16 = rows.to(torch.float16)                           # spec S2 x16 = fp16(x)
                with torch.autocast(device_type=x16.device.type, enabled=False):
                    xs = x16.index_select(1, self.keep).float() - self.off
                    xw = (xs @ self.M).half()                          # xw = fp32(fp16(xs @ M))
                    q = self.enc.f_q(xw.float()).half()                # spec S5 q = fp16(f_q(xw))
                    h = self.enc.f_t(xw.float())                       # spec S4 bits = h >= 0
                bits = (h >= 0).cpu().numpy().astype(np.uint8)
                rec = {"q": q.cpu().numpy(), "codes": np.packbits(bits, axis=1),
                       "maxabs": float(rows.abs().max().item()), "finite": bool(torch.isfinite(rows).all().item()),
                       "min_abs_h": float(h.abs().min().item())}
                if self.keep_raw:
                    rec["raw"] = x16.cpu().numpy()
                self.out.append(rec)
            return torch.zeros((res_batch_embedding.shape[0], 1), device=res_batch_embedding.device)

    return CodeCapture()


def lpt_shards(lengths, n):
    order = np.argsort(-np.asarray(lengths), kind="stable")
    load = np.zeros(n)
    shard = np.zeros(len(lengths), np.int64)
    for i in order:
        r = int(np.argmin(load))
        shard[i] = r
        load[r] += lengths[i] ** 1.3 + 200.0         # ~superlinear cost + per-protein overhead
    return shard


def devtag_of(device):
    if device.startswith("cuda"):
        return "gpu"
    if device == "mps":
        return "mps"
    return "cpu_mac" if sys.platform == "darwin" else "cpu_linux"


def cmd_embed(a):
    import torch
    import_foldmatch()
    from foldmatch.utils.model import get_residue_model
    from foldmatch.modules.chain_complete_module import ChainCompleteModule
    from foldmatch.dataset.esm_prot_from_fasta import EsmProtFromFasta
    from foldmatch.utils.fasta import iter_fasta_offsets

    device = torch.device(a.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if a.threads:
        torch.set_num_threads(a.threads)
    tag = a.devtag or devtag_of(a.device)
    # FoldMatch sets no precision flags: record the torch defaults it runs with (TF32 matmul off by default).
    flags = dict(float32_matmul_precision=torch.get_float32_matmul_precision(),
                 cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                 cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                 cuda_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
    assert not torch.backends.cuda.matmul.allow_tf32 and flags["float32_matmul_precision"] == "highest"
    deadline = float(os.environ.get("DEADLINE", "0"))
    t0 = time.time()
    res_model = get_residue_model(device=torch.device("cpu"))       # as chain_complete_inference.predict
    cap = build_capture()
    module = ChainCompleteModule(res_model=res_model, aggregator_model=cap)
    module.to(device)
    module.eval()
    t_load = time.time() - t0
    prov = dict(fm_provenance(), **esm_snapshot(), device=str(device), devtag=tag, torch=torch.__version__,
                mps_empty_cache=bool(getattr(a, "mps_empty_cache", False)),
                threads=torch.get_num_threads(), flags=flags, platform=platform.platform(), host=platform.node(),
                weights_dtype=str(next(res_model.parameters()).dtype), load_s=round(t_load, 1))
    if device.type == "cuda":
        prov["gpu"] = torch.cuda.get_device_name(device)
    import esm
    prov["esm"] = getattr(esm, "__version__", None)
    log(f"[{tag} r{a.rank}] models loaded in {t_load:.1f} s on {device}: {json.dumps(prov)}")
    tok = res_model.tokenizers.sequence
    stats = {"prov": prov, "sets": {}}
    for set_name in a.sets:
        fasta = f"{WORK}/inputs/{set_name}.fasta"
        ds = EsmProtFromFasta(fasta_file=fasta)                          # FoldMatch's reader (min_res_n 0)
        lens = np.array([ln for _, _, ln in iter_fasta_offsets(fasta)])
        names = [nm for _, nm, _ in iter_fasta_offsets(fasta)]
        assert len(lens) == len(ds)
        mine = np.flatnonzero(lpt_shards(lens, a.nranks) == a.rank) if a.nranks > 1 else np.arange(len(ds))
        pdir = f"{WORK}/emb/{tag}/{set_name}"
        os.makedirs(pdir, exist_ok=True)
        done = set()
        for f in glob.glob(f"{pdir}/*.npz"):
            with np.load(f) as z:
                done.update(z["keys"].astype(str).tolist())
        todo = [i for i in mine if names[i] not in done]
        keep_raw = set_name in (a.raw or [])
        cap.keep_raw = keep_raw
        log(f"[{tag} r{a.rank}] {set_name}: {len(mine)} of {len(ds)} unique sequences on this rank, {len(todo)} to do, "
            f"{int(lens[todo].sum()) if todo else 0:,} residues, raw={keep_raw}")
        buf = {"keys": [], "lengths": [], "q": [], "codes": [], "maxabs": [], "min_abs_h": [], "unk": [], "raw": []}
        chunk = len(glob.glob(f"{pdir}/r{a.rank}_c*.npz"))
        st = {"n": 0, "res": 0, "esm_enc_s": 0.0, "stopped": False}

        def flush():
            nonlocal chunk
            if not buf["keys"]:
                return
            blob = {"keys": np.array(buf["keys"]), "lengths": np.array(buf["lengths"], np.int64),
                    "q": np.concatenate(buf["q"]), "codes": np.concatenate(buf["codes"]),
                    "maxabs": np.array(buf["maxabs"], np.float32), "min_abs_h": np.array(buf["min_abs_h"], np.float32),
                    "unk": np.array(buf["unk"], np.int32), "prov_json": np.array(json.dumps(prov))}
            if buf["raw"]:
                blob["raw"] = np.concatenate(buf["raw"])
            fn = f"{pdir}/r{a.rank}_c{chunk:04d}.npz"
            with open(fn + ".tmp", "wb") as fh:
                np.savez(fh, **blob)
            os.replace(fn + ".tmp", fn)
            chunk += 1
            for v in buf.values():
                v.clear()

        ts = time.time()
        for n_done, i in enumerate(todo):
            if deadline and time.time() > deadline:
                st["stopped"] = True
                log(f"[{tag} r{a.rank}] {set_name}: deadline after {n_done} / {len(todo)}")
                break
            item = ds[int(i)]                                             # (ESMProtein, name)
            seq, name = item[0].sequence, item[1]
            ids = tok.encode(seq, add_special_tokens=True)
            assert len(ids) == len(seq) + 2, (name, len(ids), len(seq))
            t1 = time.time()
            cap.out.clear()
            with torch.inference_mode():
                module.predict_step([item], int(i))
            if device.type == "cuda":
                torch.cuda.synchronize()
            if device.type == "mps" and a.mps_empty_cache:
                # MPS caching-allocator growth: without this the process footprint reached 83 GB after ~1,800
                # proteins on the M1 Max (64 GB), swapping; releasing the cache does not change any value
                torch.mps.synchronize()
                torch.mps.empty_cache()
            st["esm_enc_s"] += time.time() - t1
            rec = cap.out[0]
            assert rec["finite"], name
            assert rec["q"].shape == (len(seq), 64) and rec["codes"].shape == (len(seq), 8)
            buf["keys"].append(name)
            buf["lengths"].append(len(seq))
            buf["q"].append(rec["q"])
            buf["codes"].append(rec["codes"])
            buf["maxabs"].append(rec["maxabs"])
            buf["min_abs_h"].append(rec["min_abs_h"])
            buf["unk"].append(int(sum(t == tok.unk_token_id for t in ids)))
            if keep_raw:
                buf["raw"].append(rec["raw"])
            st["n"] += 1
            st["res"] += len(seq)
            if len(buf["keys"]) >= a.flush:
                flush()
            if (n_done + 1) % 200 == 0 or n_done + 1 == len(todo):
                el = time.time() - ts
                mem = ""
                if device.type == "mps":
                    mem = (f", mps allocated {torch.mps.current_allocated_memory() / 2 ** 30:.1f} GiB, driver "
                           f"{torch.mps.driver_allocated_memory() / 2 ** 30:.1f} GiB")
                log(f"[{tag} r{a.rank}] {set_name} {n_done + 1}/{len(todo)}: {st['res'] / max(el, 1e-9):,.0f} res/s{mem}")
        flush()
        st["wall_s"] = round(time.time() - ts, 1)
        st["esm_enc_s"] = round(st["esm_enc_s"], 1)
        if device.type == "cuda":
            st["peak_gib"] = round(torch.cuda.max_memory_allocated(device) / 2 ** 30, 2)
        stats["sets"][set_name] = st
        if st["stopped"]:
            break
    os.makedirs(f"{WORK}/emb/{tag}/stats", exist_ok=True)
    jdump(stats, f"{WORK}/emb/{tag}/stats/r{a.rank}_{os.environ.get('SLURM_JOB_ID', 'local')}_{int(t0)}.json")


# =============================================================================
# merge: parts -> record-order encodings
# =============================================================================
def cmd_merge(a):
    man = load_sets()
    os.makedirs(f"{WORK}/enc", exist_ok=True)
    for set_name in a.sets:
        S = man["sets"][set_name]
        got, info, prov = {}, {}, None
        raw = {}
        for f in sorted(glob.glob(f"{WORK}/emb/{a.devtag}/{set_name}/*.npz")):
            with np.load(f) as z:
                # read every array ONCE per part (z[key] re-reads the member on each access)
                ks, Ls = z["keys"].astype(str), z["lengths"]
                off = np.concatenate([[0], np.cumsum(Ls)])
                prov = json.loads(str(z["prov_json"]))
                zq, zc, zm, zh, zu = z["q"], z["codes"], z["maxabs"], z["min_abs_h"], z["unk"]
                zr = z["raw"] if "raw" in z.files else None
                for i, k in enumerate(ks):
                    got[k] = (zq[off[i]:off[i + 1]], zc[off[i]:off[i + 1]])
                    info[k] = (float(zm[i]), float(zh[i]), int(zu[i]))
                    if zr is not None:
                        raw[k] = zr[off[i]:off[i + 1]]
        miss = [k for k in set(S["keys"]) if k not in got]
        if miss:
            log(f"{a.devtag}.{set_name}: {len(miss)} of {S['n_unique']} unique sequences missing -> not merged")
            continue
        keys = S["keys"]
        lengths = np.array([len(got[k][0]) for k in keys], np.int64)
        offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        blob = dict(ids=np.array(S["ids"]), keys=np.array(keys), lengths=lengths, offsets=offsets,
                    q=np.concatenate([got[k][0] for k in keys]), codes=np.concatenate([got[k][1] for k in keys]),
                    maxabs=np.array([info[k][0] for k in keys], np.float32),
                    min_abs_h=np.array([info[k][1] for k in keys], np.float32),
                    unk=np.array([info[k][2] for k in keys], np.int32),
                    meta_json=np.array(json.dumps(dict(set=set_name, devtag=a.devtag, prov=prov,
                                                       n_parts=len(glob.glob(f"{WORK}/emb/{a.devtag}/{set_name}/*.npz"))))))
        if raw and all(k in raw for k in keys):
            blob["raw"] = np.concatenate([raw[k] for k in keys])
        path = f"{WORK}/enc/{a.devtag}.{set_name}.npz"
        with open(path + ".tmp", "wb") as fh:
            np.savez(fh, **blob)
        os.replace(path + ".tmp", path)
        log(f"merged {a.devtag}.{set_name}: {len(keys)} records, {offsets[-1]:,} residues, max|emb| "
            f"{blob['maxabs'].max():.0f}, unk {int(blob['unk'].sum())}, raw {'raw' in blob} -> {path} ({md5(path)})")


# =============================================================================
# encodings by source, aligned to benchmark rows / columns
# =============================================================================
class Enc:
    """q [N, 64] fp16 / codes [N, 8] uint8 with offsets, for an ordered list of records."""

    def __init__(self, ids, offsets, q=None, codes=None, src=None):
        self.ids = list(ids)
        self.offsets = np.asarray(offsets, np.int64)
        self.q, self.codes, self.src = q, codes, src

    def rows(self, i, what):
        a = self.q if what == "q" else self.codes
        return a[self.offsets[i]:self.offsets[i + 1]]

    def subset(self, idx):
        idx = list(idx)
        L = np.diff(self.offsets)[idx]
        off = np.concatenate([[0], np.cumsum(L)]).astype(np.int64)
        q = np.concatenate([self.rows(i, "q") for i in idx]) if self.q is not None else None
        c = np.concatenate([self.rows(i, "codes") for i in idx]) if self.codes is not None else None
        return Enc([self.ids[i] for i in idx], off, q, c, self.src)


def research_enc(set_name, man=None):
    """Research encodings of a set in the set's record order."""
    man = man or load_sets()
    S = man["sets"][set_name]
    if set_name in ("a5q", "q200"):
        z = np.load(REF["queries"])
        ids = z["ids"].astype(str).tolist()
        e = Enc(ids, z["offsets"], q=z["q"], src="research")
        return e if set_name == "a5q" else e.subset(S["rows"])
    if set_name in ("a5t", "tsub"):
        z = np.load(REF["targets"])
        e = Enc(z["ids"].astype(str).tolist(), z["offsets"], codes=z["codes"], src="research")
        return e if set_name == "a5t" else e.subset(S["cols"])
    if set_name == "panel":
        z = np.load(REF["panel"])
        return Enc(z["ids"].astype(str).tolist(), z["offsets"], codes=z["codes"], src="research")
    if set_name == "chains":
        z = np.load(REF["chains"])
        return Enc(z["ids"].astype(str).tolist(), z["offsets"], q=z["q"], codes=z["codes"], src="research")
    if set_name == "enccheck":
        z = np.load(REF["enccheck"])
        qs, cs, L = [], [], []
        for i in S["ids"]:
            g, n = i.split("/")
            qs.append(z[f"{g}/q/{n}"])
            cs.append(z[f"{g}/code/{n}"])
            L.append(len(cs[-1]))
        return Enc(S["ids"], np.concatenate([[0], np.cumsum(L)]), q=np.concatenate(qs), codes=np.concatenate(cs),
                   src="research")
    raise KeyError(set_name)


def device_enc(src, set_name):
    """FoldMatch-path encodings of a set for a device source (merged file); q200 / tsub fall back to a5q / a5t rows."""
    man = load_sets()
    p = f"{WORK}/enc/{src}.{set_name}.npz"
    if not os.path.exists(p):
        if set_name == "q200" and os.path.exists(f"{WORK}/enc/{src}.a5q.npz"):
            return device_enc(src, "a5q").subset(man["sets"]["q200"]["rows"])
        if set_name == "tsub" and os.path.exists(f"{WORK}/enc/{src}.a5t.npz"):
            return device_enc(src, "a5t").subset(man["sets"]["tsub"]["cols"])
        return None
    z = np.load(p)
    assert z["ids"].astype(str).tolist() == man["sets"][set_name]["ids"]
    return Enc(z["ids"].astype(str).tolist(), z["offsets"], q=z["q"], codes=z["codes"], src=src)


def get_enc(src, set_name):
    return research_enc(set_name) if src == "research" else device_enc(src, set_name)


# =============================================================================
# score: research benchmark-driver arithmetic (panel/fit/scripts/cath_score.py), setting D, fwd + reversed query
# =============================================================================
_G: dict = {}


def _kernel():
    if "F" in _G:
        return _G["F"], _G["M"]
    import ctypes as _C
    if KERNEL not in sys.path:
        sys.path.insert(0, KERNEL)
    import swfast as F
    M = _C.CDLL(os.path.join(KERNEL, "libswmulti.so"))
    M.swz_ad_multi_f32.restype = None
    M.swz_ad_multi_f32.argtypes = [_C.c_void_p, _C.c_void_p, _C.c_int, _C.c_int, _C.c_void_p, _C.c_int, _C.c_int,
                                   _C.c_void_p, _C.c_void_p, _C.c_void_p, _C.c_void_p, _C.c_void_p, _C.c_void_p]
    M.sw_ad_ws_floats.restype = _C.c_size_t
    M.sw_ad_ws_floats.argtypes = [_C.c_int, _C.c_int]
    _G["F"], _G["M"] = F, M
    return F, M


R_OFF = np.array([SETTING_D[0]] * 2, np.float32)
R_GO = np.array([SETTING_D[1]] * 2, np.float32)
R_GE = np.array([SETTING_D[2]] * 2, np.float32)
R_REV = np.array([0, 1], np.int32)
CELLS = 1 << 20


def _score_block(qp, T_codes, col):
    """qp [n, 64] float32 (query_prep), T_codes [m_tot, 8], col offsets [K+1] -> [2, K] (S_f, S_r)."""
    F, M = _kernel()
    T = F.decode_unpackbits(T_codes)
    sim = np.ascontiguousarray(qp @ T.T)
    simr = np.ascontiguousarray(sim[::-1])
    n, ld = sim.shape
    K = len(col) - 1
    col = np.ascontiguousarray(col, np.int64)
    mmax = int(np.max(np.diff(col)))
    need = 2 * (n + mmax) + int(M.sw_ad_ws_floats(n, mmax))
    ws = _G.get("ws")
    if ws is None or len(ws) < need:
        ws = _G["ws"] = np.empty(need, np.float32)
    sc = np.empty((2, K), np.float32)
    M.swz_ad_multi_f32(sim.ctypes.data, simr.ctypes.data, n, ld, col.ctypes.data, K, 2, R_OFF.ctypes.data,
                       R_GO.ctypes.data, R_GE.ctypes.data, R_REV.ctypes.data, sc.ctypes.data, ws.ctypes.data)
    return sc


def _task(t):
    """t = (job, row i, target indices array) -> (job, i, idx, sf, sr)."""
    job, i, tidx = t
    J = _G["jobs"][job]
    qp = J["QP"][J["qoff"][i]:J["qoff"][i + 1]]
    toff, codes = J["toff"], J["codes"]
    n = len(qp)
    sf = np.empty(len(tidx), np.float32)
    sr = np.empty(len(tidx), np.float32)
    # contiguous runs of target indices, chunked to <= CELLS similarity floats
    budget = max(CELLS // n, 1)
    k = 0
    while k < len(tidx):
        j0 = k
        tot = 0
        while k < len(tidx) and (k == j0 or (tidx[k] == tidx[k - 1] + 1 and tot + (toff[tidx[k] + 1] - toff[tidx[k]]) <= budget)):
            tot += toff[tidx[k] + 1] - toff[tidx[k]]
            k += 1
        a, b = int(tidx[j0]), int(tidx[k - 1]) + 1
        col = toff[a:b + 1] - toff[a]
        sc = _score_block(qp, codes[toff[a]:toff[b]], col)
        sf[j0:k], sr[j0:k] = sc[0], sc[1]
    return job, i, tidx, sf, sr


def run_jobs(jobs, nwork, label):
    """jobs: list of dict(q=Enc, t=Enc, pairs=None | list of (i, j), out=path). Full matrices [nq, nt] as float32
    npy (fwd, rev) or pair vectors npz. Resumable per job (output exists -> skipped)."""
    from multiprocessing import get_context
    F, _ = _kernel()
    todo = [J for J in jobs if not os.path.exists(J["out"])]
    if not todo:
        return
    _G["jobs"] = []
    tasks = []
    for jn, J in enumerate(todo):
        q, t = J["q"], J["t"]
        JJ = dict(QP=F.query_prep(q.q), qoff=q.offsets, codes=np.ascontiguousarray(t.codes), toff=t.offsets)
        _G["jobs"].append(JJ)
        nq, nt = len(q.ids), len(t.ids)
        tl = np.diff(t.offsets)
        if J.get("pairs") is None:
            J["Sf"] = np.full((nq, nt), np.nan, np.float32)
            J["Sr"] = np.full((nq, nt), np.nan, np.float32)
            ql = np.diff(q.offsets)
            ctot = np.concatenate([[0], np.cumsum(tl)])
            for i in np.argsort(-ql, kind="stable"):
                # split the target list into ~5e7-cell tasks (long queries spread over workers)
                per = max(int(5e7 // max(ql[i], 1)), 1)
                cuts = np.searchsorted(ctot, np.arange(0, ctot[-1] + per, per))
                cuts = np.unique(np.clip(cuts, 0, nt))
                if cuts[0] != 0:
                    cuts = np.concatenate([[0], cuts])
                if cuts[-1] != nt:
                    cuts = np.concatenate([cuts, [nt]])
                for a, b in zip(cuts[:-1], cuts[1:]):
                    if b > a:
                        tasks.append((jn, int(i), np.arange(a, b)))
        else:
            P = np.asarray(J["pairs"], np.int64).reshape(-1, 2)
            J["Sf"] = np.full(len(P), np.nan, np.float32)
            J["Sr"] = np.full(len(P), np.nan, np.float32)
            J["_pos"] = {}
            for i in np.unique(P[:, 0]):
                m = np.flatnonzero(P[:, 0] == i)
                order = m[np.argsort(P[m, 1], kind="stable")]
                J["_pos"][int(i)] = order
                tasks.append((jn, int(i), P[order, 1]))
    log(f"{label}: {len(todo)} jobs, {len(tasks)} tasks, {nwork} workers")
    t0 = time.time()
    done = 0
    with get_context("fork").Pool(nwork) as pool:
        for jn, i, tidx, sf, sr in pool.imap_unordered(_task, tasks, chunksize=1):
            J = todo[jn]
            if J.get("pairs") is None:
                J["Sf"][i, tidx], J["Sr"][i, tidx] = sf, sr
            else:
                pos = J["_pos"][i]
                J["Sf"][pos], J["Sr"][pos] = sf, sr
            done += 1
            if done % max(1, len(tasks) // 20) == 0:
                log(f"  {label}: {done}/{len(tasks)} tasks, {time.time() - t0:.0f} s")
    for J in todo:
        assert np.isfinite(J["Sf"]).all() and np.isfinite(J["Sr"]).all(), J["out"]
        os.makedirs(os.path.dirname(J["out"]), exist_ok=True)
        blob = dict(Sf=J["Sf"], Sr=J["Sr"], q_ids=np.array(J["q"].ids), t_ids=np.array(J["t"].ids),
                    meta_json=np.array(json.dumps(dict(q_src=J["q"].src, t_src=J["t"].src, setting_D=SETTING_D,
                                                       kernel=KERNEL, lanes=int(_kernel()[0].VECTOR_LANES)))))
        if J.get("pairs") is not None:
            blob["pairs"] = np.asarray(J["pairs"], np.int64).reshape(-1, 2)
        with open(J["out"] + ".tmp", "wb") as fh:
            np.savez(fh, **blob)
        os.replace(J["out"] + ".tmp", J["out"])
        log(f"  wrote {J['out']}")
    _G.pop("jobs", None)
    log(f"{label}: done in {time.time() - t0:.0f} s")


def score_plan(name):
    """The scoring plan. Sources: research, gpu, cpu_linux, cpu_mac, mps (missing sources are skipped)."""
    man = load_sets()
    sc = f"{WORK}/scores"
    jobs = []
    dev = [s for s in SRCS if s != "research"]
    pairs_sub = [tuple(p) for p in man["hits"]["pairs"]]
    # q200 rows / tsub cols -> local indices
    q200_rows = man["sets"]["q200"]["rows"]
    tsub_cols = man["sets"]["tsub"]["cols"]
    rpos = {r: k for k, r in enumerate(q200_rows)}
    cpos = {c: k for k, c in enumerate(tsub_cols)}
    P_sub = [(rpos[i], cpos[j]) for i, j in pairs_sub]
    if name in ("q90", "all"):
        # Q90: every query source x shipped (research) panel codes; also x the device's own panel codes
        for qs in SRCS:
            for qset in ("a5q", "q200"):
                q = get_enc(qs, qset)
                if q is None or (qset == "q200" and qs in ("research", "gpu")):
                    continue
                for ps in SRCS:
                    p = get_enc(ps, "panel")
                    if p is None or (ps != "research" and ps != qs):
                        continue
                    jobs.append(dict(q=q, t=p, out=f"{sc}/q90.{qs}.{qset}__{ps}.panel.npz"))
    if name in ("a5", "all"):
        q, t = get_enc("gpu", "a5q"), get_enc("gpu", "a5t")
        if q is not None and t is not None:
            jobs.append(dict(q=q, t=t, out=f"{sc}/full.gpu.a5q__gpu.a5t.npz"))
    if name in ("mixed", "all"):
        t = get_enc("gpu", "a5t")
        for qs in dev:
            if qs == "gpu":
                continue
            q = get_enc(qs, "q200")
            if q is not None and t is not None:
                jobs.append(dict(q=q, t=t, out=f"{sc}/full.{qs}.q200__gpu.a5t.npz"))
    if name in ("pairs", "all"):
        # research-hit sample pairs (q200 x tsub): every (query source, target source) combination available
        for qs in SRCS:
            q = get_enc(qs, "q200")
            if q is None:
                continue
            for ts in SRCS:
                t = get_enc(ts, "tsub")
                if t is None:
                    continue
                jobs.append(dict(q=q, t=t, pairs=P_sub, out=f"{sc}/pairs.{qs}.q200__{ts}.tsub.npz"))
    if name in ("qonly", "all"):
        # query-only effect: device queries x research codes on ALL research E<=1 hits of the 200 queries
        allhits = man["hits"].get("all_pairs")
        if allhits:
            t = research_enc("a5t")
            for qs in SRCS:
                q = get_enc(qs, "q200")
                if q is None:
                    continue
                P = [(rpos[i], j) for i, j in allhits]
                jobs.append(dict(q=q, t=t, pairs=P, out=f"{sc}/hits.{qs}.q200__research.a5t.npz"))
    return jobs


def cmd_score(a):
    for plan in a.plan:
        run_jobs(score_plan(plan), a.nwork, plan)


# =============================================================================
# chain-set SW F1 (chains/chain_eval.py, harness DP, setting D)
# =============================================================================
def chain_fns(qsrc, tsrc):
    qe, te = get_enc(qsrc, "chains"), get_enc(tsrc, "chains")
    if qe is None or te is None:
        return None
    qi = {c: k for k, c in enumerate(qe.ids)}
    ti = {c: k for k, c in enumerate(te.ids)}

    def qf(c, e):
        k = qi[c]
        r = qe.rows(k, "q")
        assert len(r) == len(e)
        return r

    def tf(c, e):
        k = ti[c]
        return np.unpackbits(te.rows(k, "codes"), axis=1)[:, :64].astype(np.float32) * 2 - 1
    return qf, tf


def cmd_chainf1(a):
    sys.path.insert(0, f"{EMB}/chains")
    import chain_eval as CE
    out = f"{WORK}/chainf1"
    os.makedirs(out, exist_ok=True)
    for spec in a.srcs:
        qs, ts = spec.split("@") if "@" in spec else (spec, spec)
        path = f"{out}/{qs}__{ts}.npz"
        if os.path.exists(path):
            continue
        fns = chain_fns(qs, ts)
        if fns is None:
            log(f"chainf1 {qs}__{ts}: encodings missing, skipped")
            continue
        CE.evaluate(f"{qs}__{ts}", query_fn=fns[0], target_fn=fns[1], sw=SETTING_D, out_dir=out, workers=a.workers,
                    verbose=False, meta=dict(q_src=qs, t_src=ts))
        log(f"chainf1 {qs}__{ts} -> {path}")


# =============================================================================
# analyze
# =============================================================================
def bit_agreement(a_codes, b_codes):
    ba, bb = np.unpackbits(a_codes, axis=1), np.unpackbits(b_codes, axis=1)
    eq = ba == bb
    return dict(bits_total=int(eq.size), bits_equal=int(eq.sum()), agreement=float(eq.mean()),
                residues=int(len(eq)), residues_code_identical=float(eq.all(1).mean()))


def per_item_bits(ea, eb):
    ag = []
    for k in range(len(ea.ids)):
        x, y = ea.rows(k, "codes"), eb.rows(k, "codes")
        ag.append(float((np.unpackbits(x, axis=1) == np.unpackbits(y, axis=1)).mean()))
    ag = np.array(ag)
    return dict(items=len(ag), items_identical=int((ag == 1.0).sum()), item_min=float(ag.min()),
                item_p5=float(np.percentile(ag, 5)), item_median=float(np.median(ag)))


def q_cosine(qa, qb):
    x, y = qa.astype(np.float64), qb.astype(np.float64)
    c = (x * y).sum(1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1))
    return dict(residues=int(len(c)), min=float(c.min()), p0_1=float(np.percentile(c, 0.1)),
                p1=float(np.percentile(c, 1)), median=float(np.median(c)), mean=float(c.mean()),
                frac_ge_0_99=float((c >= 0.99).mean()), frac_ge_0_999=float((c >= 0.999).mean()),
                q_bitwise_equal_rows=float((qa == qb).all(1).mean()))


def dstats(d):
    d = np.abs(np.asarray(d, np.float64))
    if d.size == 0:
        return dict(n=0)
    return dict(n=int(d.size), median=float(np.median(d)), p90=float(np.percentile(d, 90)),
                p95=float(np.percentile(d, 95)), p99=float(np.percentile(d, 99)), max=float(d.max()),
                mean=float(d.mean()), frac_gt_0_5=float((d > 0.5).mean()), frac_gt_1=float((d > 1).mean()))


def load_score(name):
    p = f"{WORK}/scores/{name}.npz"
    if not os.path.exists(p):
        return None
    z = np.load(p)
    return dict(S=z["Sf"].astype(np.float64) - z["Sr"].astype(np.float64), Sf=z["Sf"], Sr=z["Sr"],
                q_ids=z["q_ids"].astype(str), t_ids=z["t_ids"].astype(str),
                pairs=z["pairs"] if "pairs" in z.files else None, meta=json.loads(str(z["meta_json"])))


def q90_of(qsrc, rows, panel="research"):
    """Q90 of the benchmark query rows `rows` (indices into the 1,465) for a query source vs a panel code source."""
    man = load_sets()
    q200 = man["sets"]["q200"]["rows"]
    for qset in ("a5q", "q200"):
        d = load_score(f"q90.{qsrc}.{qset}__{panel}.panel")
        if d is None:
            continue
        q90 = np.quantile(d["S"], 0.9, axis=1)
        if qset == "a5q":
            return q90[np.asarray(rows)]
        pos = {r: k for k, r in enumerate(q200)}
        if all(r in pos for r in rows):
            return q90[[pos[r] for r in rows]]
    return None


def hit_agreement(L_ref, L_x, xs=(1.0, 1e-3)):
    out = {}
    for x in xs:
        a, b = L_ref <= math.log(x), L_x <= math.log(x)
        both, ra, xb = int((a & b).sum()), int(a.sum()), int(b.sum())
        out[str(x)] = dict(ref=ra, config=xb, both=both, ref_only=ra - both, config_only=xb - both,
                           jaccard=float(both / max(ra + xb - both, 1)))
    return out


def dlnE_block(L_ref, L_x, mask_ref_hits):
    """|dlnE| statistics on research hits (mask) and on the union of research and config hits (E <= 1)."""
    union = mask_ref_hits | (L_x <= 0.0)
    d_ref = (L_x - L_ref)[mask_ref_hits]
    d_uni = (L_x - L_ref)[union]
    return dict(research_hits=dstats(d_ref), union_hits=dstats(d_uni),
                signed_median_research_hits=float(np.median(d_ref)) if d_ref.size else None)


def lnE_bins(L_ref, d, edges=(-np.inf, -23.03, -6.91, -2.30, 0.0)):
    out = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (L_ref > lo) & (L_ref <= hi)
        out[f"lnE_ref in ({lo:.2f}, {hi:.2f}]"] = dstats(d[m])
    return out


def cmd_analyze(a):
    man = load_sets()
    BL = BenchLabels()
    tab_all = ship_table()
    tab = tab_all["all"]
    N = BL.N_ann
    q200 = np.asarray(man["sets"]["q200"]["rows"])
    tsub = np.asarray(man["sets"]["tsub"]["cols"])
    rpos = {r: k for k, r in enumerate(q200)}
    cpos = {c: k for k, c in enumerate(tsub)}
    P_sub = np.array([(rpos[i], cpos[j]) for i, j in man["hits"]["pairs"]], np.int64)
    P_all = np.array([(rpos[i], j) for i, j in man["hits"]["all_pairs"]], np.int64)
    devs = [s for s in SRCS if s != "research" and os.path.exists(f"{WORK}/enc/{s}.enccheck.npz")]
    out = {"meta": dict(
        what="P0.5 ESM3 parity (F6) + A5 re-embedding: FoldMatch ESM3 residue path + research zcw50_clean / M8_s0 "
             "encoders vs the research encodings",
        created=time.strftime("%Y-%m-%d %H:%M:%S %Z"), host=platform.node(), script=os.path.abspath(__file__),
        script_md5=md5(os.path.abspath(__file__)), work=WORK, devices=devs,
        inputs=dict(sets_json_md5=md5(f"{WORK}/inputs/sets.json"),
                    sets={k: {kk: v[kk] for kk in ("n_records", "n_unique", "residues", "fasta_md5",
                                                   "normalisation_changed", "n_nonstandard_records", "source")}
                          for k, v in man["sets"].items()},
                    q200=dict(seed=Q200_SEED, alloc=Q200_ALLOC, edges=EDGES[:-1]),
                    hits={k: v for k, v in man["hits"].items() if k not in ("pairs", "all_pairs")}),
        research=dict(model=MODEL, model_md5=md5(MODEL), zcw50_clean=ZCW, zcw50_clean_md5=md5(ZCW),
                      sw_common_md5=md5(f"{SWM}/sw_common.py"), ship_final=SHIP, ship_final_md5=md5(SHIP),
                      ship_key=SHIP_KEY, q90_shipped=Q90_SHIPPED, labels_md5=md5(LABELS),
                      refs={k: (v, md5(v)) for k, v in REF.items() if os.path.isfile(v) and k not in ("val", "test")}),
        evalue=dict(formula="lnE = lnK(m) + ln m + ln N - lambda(m) (S - Q90); lambda, lnK = np.interp(ln m, ln centre, "
                            "table) (ship_final 'all'); S = float64(S_f) - float64(S_r), setting D (1.0; 5/0.5), "
                            "S_r = query rows reversed; Q90 = np.quantile(S(query, 1,000 shipped panel codes), 0.9); "
                            "m = query length; N = annotated residues of the 50,000 benchmark targets",
                    N=N, setting_D=SETTING_D),
        definitions=dict(
            code_bits="fraction of equal code bits (64 per residue) between the FoldMatch-path codes and the research "
                      "codes of the same sequence",
            q_cosine="per-residue cosine of the fp16 query encodings (float64 math)",
            dQ90="Q90(FoldMatch-path q vs shipped research panel codes) - Q90(research q vs the same codes)",
            dlnE="lnE(config) - lnE(research) per (query, target) pair; config = (query-encoding source x target-code "
                 "source), each with its own Q90 (its q vs the shipped panel); 'research_hits' = pairs with research "
                 "E <= 1, 'union_hits' = research or config E <= 1 (full matrices only)",
            P_sub=f"{len(P_sub)} sampled research E<=1 hit pairs of the 200 queries (<= {HITS_PER_QUERY} per query), "
                  f"{len(tsub)} distinct targets, embedded on every device",
            P_all=f"all {len(P_all)} research E<=1 hit pairs of the 200 queries (needs target codes of every "
                  "benchmark target: research or GPU)",
            chain_f1="chains/chain_eval.py evaluate(): harness rowcol z + float64 SW, setting D; mean F1 vs the "
                     "US-align primary reference over the 582 similar pairs (unrelated controls excluded)"))}
    for d in devs:
        z = np.load(f"{WORK}/enc/{d}.enccheck.npz")
        out["meta"].setdefault("device_provenance", {})[d] = json.loads(str(z["meta_json"]))["prov"]

    # ---------------------------------------------------------------- code bits / q cosine vs research
    cb, qc, per_item = {}, {}, {}
    for d in devs:
        cb[d], qc[d], per_item[d] = {}, {}, {}
        pool_eq = pool_tot = 0
        for s in ("panel", "enccheck", "chains", "tsub", "q200", "a5t", "a5q"):
            e = device_enc(d, s)
            if e is None:
                continue
            r = research_enc(s)
            if r.codes is not None:
                cb[d][s] = bit_agreement(r.codes, e.codes)
                per_item[d][s] = per_item_bits(r, e)
                if s in ("panel", "enccheck", "chains", "tsub"):
                    pool_eq += cb[d][s]["bits_equal"]
                    pool_tot += cb[d][s]["bits_total"]
            if r.q is not None:
                qc[d][s] = q_cosine(r.q, e.q)
        cb[d]["pooled_panel_enccheck_chains_tsub"] = dict(bits_total=pool_tot, bits_equal=pool_eq,
                                                          agreement=pool_eq / max(pool_tot, 1))
        log(f"code bits {d}: " + ", ".join(f"{s} {v['agreement']:.5f}" for s, v in cb[d].items()))
    out["code_bits_vs_research"] = cb
    out["code_bits_per_item_vs_research"] = per_item
    out["q_cosine_vs_research"] = qc

    # ---------------------------------------------------------------- cross-device agreement
    xd = {}
    for d1, d2 in (("cpu_mac", "cpu_linux"), ("gpu", "cpu_linux"), ("gpu", "cpu_mac"), ("mps", "cpu_mac"),
                   ("mps", "cpu_linux"), ("gpu", "mps")):
        if d1 not in devs or d2 not in devs:
            continue
        k = f"{d1}_vs_{d2}"
        xd[k] = {}
        eq = tot = 0
        for s in ("panel", "enccheck", "chains", "tsub", "q200"):
            e1, e2 = device_enc(d1, s), device_enc(d2, s)
            if e1 is None or e2 is None:
                continue
            b = bit_agreement(e1.codes, e2.codes)
            xd[k][s] = dict(code_bits=b, per_item=per_item_bits(e1, e2), q_cosine=q_cosine(e1.q, e2.q))
            if s != "q200":
                eq += b["bits_equal"]
                tot += b["bits_total"]
        xd[k]["pooled_code_bits_panel_enccheck_chains_tsub"] = dict(bits_total=tot, bits_equal=eq,
                                                                    agreement=eq / max(tot, 1))
        log(f"cross-device {k}: pooled code bits {eq / max(tot, 1):.6f}")
    out["cross_device"] = xd

    # ---------------------------------------------------------------- raw embeddings (enccheck vs the research store rows)
    emb = {}
    if f"{EMB}/chains" not in sys.path:
        sys.path.insert(0, f"{EMB}/chains")
    import chain_eval as CE
    store = {}
    for g, path, idkey in (("val", REF["val"], "domain_ids"), ("test", REF["test"], "domain_ids"),
                           ("chains", REF["chain_emb"], "chain_ids")):
        z = np.load(path)
        try:
            e = CE._npz_memmap(path, "emb")             # uncompressed member -> memmap (no 1.3 GB read)
        except AssertionError:
            e = z["emb"]
        store[g] = (e, z["offsets"], {x: k for k, x in enumerate(z[idkey].astype(str))})

    def store_rows(i):
        g, n = i.split("/")
        e, off, ix = store[g]
        k = ix[n]
        return np.asarray(e[off[k]:off[k + 1]])
    ref_rows = np.concatenate([store_rows(i) for i in man["sets"]["enccheck"]["ids"]]).astype(np.float64)
    raws = {}
    for d in devs:
        z = np.load(f"{WORK}/enc/{d}.enccheck.npz")
        if "raw" in z.files:
            raws[d] = z["raw"].astype(np.float64)
            c = (raws[d] * ref_rows).sum(1) / (np.linalg.norm(raws[d], axis=1) * np.linalg.norm(ref_rows, axis=1))
            emb[f"{d}_vs_research_store"] = dict(residues=int(len(c)), min=float(c.min()), p1=float(np.percentile(c, 1)),
                                                 median=float(np.median(c)), mean=float(c.mean()),
                                                 rel_frobenius=float(np.linalg.norm(raws[d] - ref_rows)
                                                                     / np.linalg.norm(ref_rows)))
    for d1, d2 in (("cpu_mac", "cpu_linux"), ("gpu", "cpu_linux"), ("mps", "cpu_mac")):
        if d1 in raws and d2 in raws:
            c = (raws[d1] * raws[d2]).sum(1) / (np.linalg.norm(raws[d1], axis=1) * np.linalg.norm(raws[d2], axis=1))
            emb[f"{d1}_vs_{d2}"] = dict(min=float(c.min()), median=float(np.median(c)), mean=float(c.mean()),
                                        rows_bit_identical=float((raws[d1] == raws[d2]).all(1).mean()),
                                        rel_frobenius=float(np.linalg.norm(raws[d1] - raws[d2]) / np.linalg.norm(raws[d2])))
    out["embedding_enccheck"] = emb

    # ---------------------------------------------------------------- encoder device effect: device fp16 rows re-encoded
    # on the CPU (fp32, per protein) vs the codes / q the device's own encoder pass produced
    try:
        import torch
        cap = build_capture()
        enc_eff = {}
        for d in devs:
            for s in ("enccheck", "q200", "a5q"):
                p = f"{WORK}/enc/{d}.{s}.npz"
                if not os.path.exists(p) or (s == "a5q" and d != "gpu"):
                    continue
                z = np.load(p)
                if "raw" not in z.files:
                    continue
                off, raw = z["offsets"], z["raw"]
                nb = ne = 0
                qeq = []
                for k in range(len(off) - 1):
                    x = torch.from_numpy(np.ascontiguousarray(raw[off[k]:off[k + 1]]))
                    with torch.no_grad():
                        xs = x.index_select(1, cap.keep).float() - cap.off
                        xw = (xs @ cap.M).half()
                        q = cap.enc.f_q(xw.float()).half().numpy()
                        c = np.packbits((cap.enc.f_t(xw.float()) >= 0).numpy().astype(np.uint8), axis=1)
                    b1, b2 = np.unpackbits(c, axis=1), np.unpackbits(z["codes"][off[k]:off[k + 1]], axis=1)
                    nb += b1.size
                    ne += int((b1 == b2).sum())
                    qeq.append((q == z["q"][off[k]:off[k + 1]]).all(1))
                enc_eff[f"{d}.{s}"] = dict(bits_total=nb, bits_equal=ne, agreement=ne / max(nb, 1),
                                           q_rows_bitwise_equal=float(np.concatenate(qeq).mean()))
        out["encoder_device_effect_cpu_reencode"] = enc_eff
    except Exception as ex:          # diagnostic only
        out["encoder_device_effect_cpu_reencode"] = {"error": repr(ex)}

    # ---------------------------------------------------------------- dQ90
    rows_all = np.arange(BL.nq)
    q90R = q90_of("research", rows_all)
    shipped = np.load(Q90_SHIPPED)
    dq = {"research_recomputed_vs_bench_q90_shipped": dstats(q90R - shipped) if q90R is not None else None}
    for d in devs:
        for rows, lab in ((q200, "q200"), (rows_all, "a5q")):
            x = q90_of(d, rows)
            if x is None:
                continue
            dq[f"{d}.{lab}"] = dict(dstats(x - q90R[rows]), signed_median=float(np.median(x - q90R[rows])))
            xp = q90_of(d, rows, panel=d)
            if xp is not None:
                dq[f"{d}.{lab}.own_panel_codes"] = dict(dstats(xp - q90R[rows]),
                                                        signed_median=float(np.median(xp - q90R[rows])))
    out["dQ90"] = dq
    # largest |dQ90| queries of the GPU A5 queries, with their per-residue q cosine to the research encodings
    qg = q90_of("gpu", rows_all)
    if qg is not None:
        rq, gq = research_enc("a5q"), device_enc("gpu", "a5q")
        top = []
        for i in np.argsort(-np.abs(qg - q90R))[:5]:
            c = q_cosine(rq.rows(i, "q"), gq.rows(i, "q"))
            top.append(dict(id=rq.ids[i], length=int(BL.qlen[i]), dQ90=float(qg[i] - q90R[i]), q90_research=float(q90R[i]),
                            q_cos_median=c["median"], q_cos_min=c["min"]))
        out["dQ90_top_gpu_a5q"] = top

    # ---------------------------------------------------------------- dlnE
    dl = {"P_sub": {}, "P_all": {}, "full_q200": {}, "full_a5": {}}
    m200 = BL.qlen[q200]
    ref_sub = load_score("pairs.research.q200__research.tsub")
    Lref_sub = None
    if ref_sub is not None:
        assert np.array_equal(ref_sub["pairs"], P_sub)
        S_bench = research_S(q200)
        S_bench_sub = S_bench[P_sub[:, 0], tsub[P_sub[:, 1]]]
        dl["research_recomputed_vs_research_matrix_P_sub"] = dstats(ref_sub["S"] - S_bench_sub)
        Lref_sub = ln_evalue(tab, m200[P_sub[:, 0]], ref_sub["S"][:, None], q90R[q200][P_sub[:, 0]], N)[:, 0]
        # config x config on P_sub
        for qs in SRCS:
            for ts in SRCS:
                if qs == ts == "research":
                    continue
                d = load_score(f"pairs.{qs}.q200__{ts}.tsub")
                if d is None:
                    continue
                q90 = q90_of(qs, q200)
                if q90 is None:
                    continue
                Lx = ln_evalue(tab, m200[P_sub[:, 0]], d["S"][:, None], q90[P_sub[:, 0]], N)[:, 0]
                dd = Lx - Lref_sub
                dl["P_sub"][f"{qs}__{ts}"] = dict(dstats(dd), signed_median=float(np.median(dd)),
                                                 by_research_lnE=lnE_bins(Lref_sub, dd),
                                                 hits_E_le_1_config=int((Lx <= 0).sum()),
                                                 hits_E_le_1e3_ref=int((Lref_sub <= math.log(1e-3)).sum()),
                                                 hits_E_le_1e3_config=int((Lx <= math.log(1e-3)).sum()))
    # research lnE for every (q200, target)
    Lref200 = ln_evalue(tab, m200, research_S(q200), q90R[q200], N)
    ref_hit200 = Lref200 <= 0.0
    assert int(ref_hit200.sum()) == len(P_all)
    # query-only effect on all research hits: device q x research codes
    for qs in SRCS:
        d = load_score(f"hits.{qs}.q200__research.a5t")
        if d is None:
            continue
        q90 = q90R[q200] if qs == "research" else q90_of(qs, q200)
        Lx = ln_evalue(tab, m200[P_all[:, 0]], d["S"][:, None], q90[P_all[:, 0]], N)[:, 0]
        Lr = Lref200[P_all[:, 0], P_all[:, 1]]
        dl["P_all"][f"{qs}__research"] = dict(dstats(Lx - Lr), signed_median=float(np.median(Lx - Lr)))
    # full matrices: GPU x GPU (a5) and device queries x GPU codes (q200)
    g = load_score("full.gpu.a5q__gpu.a5t")
    if g is not None:
        q90G = q90_of("gpu", rows_all)
        LG = ln_evalue(tab, BL.qlen, g["S"], q90G, N)
        LR = ln_evalue(tab, BL.qlen, research_S(), q90R, N)
        rh = LR <= 0.0
        dl["full_a5"]["gpu__gpu"] = dict(dlnE_block(LR, LG, rh), hit_agreement=hit_agreement(LR, LG),
                                        by_research_lnE=lnE_bins(LR[rh], (LG - LR)[rh]))
        LG200 = LG[q200]
        dl["full_q200"]["gpu__gpu"] = dict(dlnE_block(Lref200, LG200, ref_hit200),
                                          hit_agreement=hit_agreement(Lref200, LG200))
        dS = g["S"] - research_S()
        out["dS_full_a5_gpu"] = dict(all_pairs=dstats(dS.ravel()[::97]), research_hits=dstats(dS[rh]))
        # A5 preview: FP ratio / sensitivity with FoldMatch GPU encodings (bench_validate 'own' NEW protocol)
        out["a5_preview"] = a5_preview(BL, tab_all, g["S"], q90G, research_S(), q90R)
    for qs in SRCS:
        if qs in ("research", "gpu"):
            continue
        d = load_score(f"full.{qs}.q200__gpu.a5t")
        if d is None:
            continue
        Lx = ln_evalue(tab, m200, d["S"], q90_of(qs, q200), N)
        dl["full_q200"][f"{qs}__gpu"] = dict(dlnE_block(Lref200, Lx, ref_hit200),
                                            hit_agreement=hit_agreement(Lref200, Lx))
        if g is not None:
            dd = (Lx - LG200)[ref_hit200 | (LG200 <= 0)]
            dl["full_q200"][f"{qs}__gpu_vs_gpu__gpu"] = dstats(dd)
    out["dlnE"] = dl

    # ---------------------------------------------------------------- chain-set SW F1
    out["chain_f1"] = chain_f1_summary()

    # ---------------------------------------------------------------- provisional thresholds (plan section 4 P0.5)
    out["provisional_threshold_check"] = threshold_check(out, devs)
    jdump(out, f"{WORK}/esm3_parity.json")
    log(f"wrote {WORK}/esm3_parity.json ({md5(f'{WORK}/esm3_parity.json')})")


def a5_preview(BL, tab_all, S_g, q90_g, S_r, q90_r):
    idx = np.arange(BL.nq)
    ev, od = idx[idx % 2 == 0], idx[idx % 2 == 1]
    out = {}
    for name, S, q90 in (("research", S_r, q90_r), ("gpu", S_g, q90_g)):
        o = {}
        c_union = {}
        for sp, rows in (("main", od), ("swap", ev)):
            L = ln_evalue(tab_all[sp], BL.qlen[rows], S[rows], q90[rows], BL.N_ann)
            REAL = BL.REAL[rows]
            o[sp] = {}
            for x in (0.01, 0.1, 1.0):
                c = ((L <= math.log(x)) & REAL).sum(1)
                o[sp][str(x)] = dict(fp=int(c.sum()), ratio=float(c.sum() / (len(rows) * x)))
                c_union.setdefault(str(x), 0)
                c_union[str(x)] += int(c.sum())
            Ls = L + math.log(BL.N_all) - math.log(BL.N_ann)
            TP = BL.TP[rows]
            o[sp]["sens_1e-3"] = float(((Ls <= math.log(1e-3)) & TP).sum() / TP.sum())
        o["heldout_union"] = {x: dict(fp=v, ratio=v / (BL.nq * float(x))) for x, v in c_union.items()}
        out[name] = o
    return out


def chain_f1_summary():
    d = f"{WORK}/chainf1"
    if not os.path.isdir(d):
        return None
    sys.path.insert(0, f"{EMB}/chains")
    import chain_eval as CE
    S = CE.load()
    sim = np.array([p["kind"] != "unrelated" for p in S.pairs])
    ref = CE.load_result(CHAIN_REF_D)
    fref = np.nan_to_num(ref["fields"]["f1"][sim])
    out = {"research_M8_s0_D_mean_f1": float(fref.mean()), "n_similar": int(sim.sum()), "configs": {}}
    for p in sorted(glob.glob(f"{d}/*.npz")):
        r = CE.load_result(p)
        f = np.nan_to_num(r["fields"]["f1"][sim])
        same = [((x is None and y is None) or (x is not None and y is not None and np.array_equal(x, y)))
                for x, y in zip(r["alignments"], ref["alignments"])]
        pdl = CE.paired_delta(r, ref)["ALL similar"]
        pdc = CE.paired_delta(r, ref, cluster=True)["ALL similar"]
        out["configs"][r["name"]] = dict(mean_f1=float(f.mean()), dF1=float(f.mean() - fref.mean()),
                                         dF1_ci_pairs=[pdl[1], pdl[2]], dF1_ci_cluster=[pdc[1], pdc[2]],
                                         alignments_identical=float(np.mean(same)),
                                         max_abs_pair_dF1=float(np.abs(f - fref).max()),
                                         unrelated_mean_aligned=float(np.nanmean(r["fields"]["sw_n"][~sim])))
    return out


PROVISIONAL = {"code_bits": {"gpu": 0.98, "cpu": 0.97}, "dlnE_median": 0.1, "dlnE_p95": 0.5, "dF1": 0.01,
               "cpu_mac_vs_cpu_linux_bits": 0.995}


def threshold_check(out, devs):
    res = {}
    for d in devs:
        kind = "gpu" if d == "gpu" else "cpu"
        cb = out["code_bits_vs_research"][d]["pooled_panel_enccheck_chains_tsub"]["agreement"]
        r = {"code_bits": dict(value=cb, threshold=PROVISIONAL["code_bits"][kind],
                               passed=bool(cb >= PROVISIONAL["code_bits"][kind]))}
        dd = out["dlnE"]["P_sub"].get(f"{d}__{d}")
        if dd:
            r["dlnE_P_sub"] = dict(median=dd["median"], p95=dd["p95"],
                                   passed=bool(dd["median"] <= PROVISIONAL["dlnE_median"]
                                               and dd["p95"] <= PROVISIONAL["dlnE_p95"]))
        cf = (out.get("chain_f1") or {}).get("configs", {}).get(f"{d}__{d}")
        if cf:
            r["dF1"] = dict(value=cf["dF1"], passed=bool(abs(cf["dF1"]) <= PROVISIONAL["dF1"]))
        res[d] = r
    for k, v in out["dlnE"]["P_sub"].items():
        qs, ts = k.split("__")
        if qs != ts and "research" not in (qs, ts):
            res[f"mixed {k}"] = dict(median=v["median"], p95=v["p95"],
                                     passed=bool(v["median"] <= PROVISIONAL["dlnE_median"] and v["p95"] <= PROVISIONAL["dlnE_p95"]))
    x = out["cross_device"].get("cpu_mac_vs_cpu_linux")
    if x:
        v = x["pooled_code_bits_panel_enccheck_chains_tsub"]["agreement"]
        res["cpu_mac_vs_cpu_linux_bits"] = dict(value=v, threshold=PROVISIONAL["cpu_mac_vs_cpu_linux_bits"],
                                                passed=bool(v >= PROVISIONAL["cpu_mac_vs_cpu_linux_bits"]))
    return res


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


# A4 gates: the decisions (value + reason) are fixed here; `thresholds` fills in the P0.5 observations.
A4_BITS = 0.990
A4_DLNE_MEDIAN, A4_DLNE_P95 = 0.1, 0.5
A4_DF1 = 0.01
A4_XCPU_BITS = 0.999


def cmd_thresholds(a):
    P = json.load(open(a.parity))
    man = load_sets()
    cb = P["code_bits_vs_research"]
    devs = [d for d in ("gpu", "cpu_linux", "cpu_mac", "mps") if d in cb]
    pooled = {d: cb[d]["pooled_panel_enccheck_chains_tsub"]["agreement"] for d in devs}
    per_set = {d: {s: v["agreement"] for s, v in cb[d].items() if s != "pooled_panel_enccheck_chains_tsub"} for d in devs}
    dl = {k: dict(median=v["median"], p95=v["p95"], p99=v["p99"], max=v["max"])
          for k, v in P["dlnE"]["P_sub"].items() if "research" not in k}
    cf = {k: dict(mean_f1=v["mean_f1"], dF1=v["dF1"], ci_pairs=v["dF1_ci_pairs"])
          for k, v in P["chain_f1"]["configs"].items() if k != "research__research"}
    xd = {k: v["pooled_code_bits_panel_enccheck_chains_tsub"]["agreement"] for k, v in P["cross_device"].items()}
    xitem = {k: min(vv["per_item"]["item_min"] for s, vv in v.items() if s != "pooled_code_bits_panel_enccheck_chains_tsub")
             for k, v in P["cross_device"].items()}
    tsub = man["sets"]["tsub"]
    Pd = P["dlnE"]["P_sub"]
    rng = lambda keys, f: (min(Pd[k][f] for k in keys), max(Pd[k][f] for k in keys))  # noqa: E731
    dd_keys = [k for k in Pd if "research" not in k]
    rq_keys = [f"research__{d}" for d in devs if f"research__{d}" in Pd]
    qr_keys = [f"{d}__research" for d in devs if f"{d}__research" in Pd]
    m_dd, p_dd, p99_dd, x_dd = rng(dd_keys, "median"), rng(dd_keys, "p95"), rng(dd_keys, "p99"), rng(dd_keys, "max")
    m_rq, p_rq, m_qr, p_qr = rng(rq_keys, "median"), rng(rq_keys, "p95"), rng(qr_keys, "median"), rng(qr_keys, "p95")
    dF1max = max(abs(v["dF1"]) for v in cf.values())
    ci_lo, ci_hi = min(v["ci_pairs"][0] for v in cf.values()), max(v["ci_pairs"][1] for v in cf.values())
    out = {
        "name": "A4_THRESHOLDS", "format_version": 1, "system": "esm-codes-v1",
        "calibration_version": "foldmatch-evalue-M8_s0-D_fmr-pnlU50s1000-M3t999-v1",
        "plan": "FoldMatch esm-codes integration plan FINAL v1, section 4 P0.5 (thresholds) and section 8 P8 A4",
        "status": "PRE-REGISTERED from the P0.5 evidence. Commit before A4 runs; the A4 output must record this file's "
                  "sha256. Any change after A4 has started is a new file version with a stated reason, never an "
                  "in-place edit.",
        "created": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "evidence": {
            "esm3_parity_json": dict(path=os.path.abspath(a.parity), nersc="/pscratch/sd/j/jsegura/embcomp/p0/parity/esm3_parity.json",
                                     md5=md5(a.parity), sha256=sha256(a.parity)),
            "script": dict(path=os.path.abspath(__file__), md5=md5(os.path.abspath(__file__))),
            "inputs_sets_json": dict(path=f"{WORK}/inputs/sets.json", md5=md5(f"{WORK}/inputs/sets.json"),
                                     sha256=sha256(f"{WORK}/inputs/sets.json")),
            "foldmatch": P["meta"]["device_provenance"]["gpu"].get("commit") if "gpu" in P["meta"].get("device_provenance", {}) else None,
            "archive": "/global/homes/j/jsegura/esm_codes_archive/p0/parity (NERSC home; inputs/sets.json, "
                       "esm3_parity.json, compact encodings of every device incl. the A5 GPU encodings, MD5SUMS)",
        },
        "protocol": {
            "what": "A4 re-runs the P0.5 measurement through FoldMatch's production implementation (ESM3 residue path "
                    "of build/query, the P2 encoder module, the P3 kernel, the P4 statistics) on each device class it "
                    "claims, and compares with the RESEARCH encodings exactly as P0.5 did.",
            "inputs": {k: dict(n_records=v["n_records"], residues=v["residues"], fasta_md5=v["fasta_md5"])
                       for k, v in man["sets"].items() if k in ("panel", "enccheck", "chains", "q200", "tsub")},
            "inputs_note": "sets.json (sha256 above) fixes every id, sequence key (sha1 of the normalised sequence) and "
                           "order; q200 = 200 benchmark queries stratified by M3 length bin (34/34/33/33/33/33, "
                           f"default_rng({Q200_SEED})); tsub = the {tsub['n_records']} targets of P_sub",
            "P_sub": dict(n_pairs=man["hits"]["n_pairs"], n_targets=man["hits"]["n_targets"],
                          definition=man["hits"]["definition"],
                          pairs_sha256=hashlib.sha256(json.dumps(man["hits"]["pairs"]).encode()).hexdigest()),
            "reference": "research encodings: queries evalue/encode/out/queries.L2m_B8_s0.npz (md5 e821497f...), targets "
                         "targets.L2m_B8_s0.npz (cfee06cd...), panel shipped_panel_codes.npz (15cc1c7b...), chains "
                         "sw_mlp/enc/L2m_B8_s0.chains.npz (55198660...), enc_check sw_mlp/enc_check/L2m_B8_s0.npz; research "
                         "lnE from the research S_D matrices (evalue/score/out) and bench_q90_shipped.npy (P0.5 "
                         "recomputation: Q90 identical, P_sub S max |d| 1e-4)",
            "evalue": P["meta"]["evalue"],
            "metrics": {
                "code_bits": "fraction of equal code bits vs the research codes, POOLED over panel + enccheck + chains + "
                             "tsub (783,823 residues, 50,164,672 bits)",
                "dlnE": "lnE(config) - lnE(research) on the P_sub pairs; config = (query-encoding device, target-code "
                        "device); each config's Q90 = its own q vs the SHIPPED research panel codes; |dlnE| median and "
                        "P95 over the pairs",
                "dF1": "mean SW F1 (setting D, chain_eval.prf vs the US-align primary reference) over the 582 similar "
                       "chain-set pairs minus the research M8_s0 setting-D value 0.613393 "
                       "(evalue/f1_settings/results/chains/M8_s0_D.npz); two-sided",
                "xcpu_bits": "code-bit agreement between two CPU fp32 runs on different machines / OS / BLAS, pooled "
                             "over the compared residues (the F2c cross-machine threshold)"},
            "device_classes": {
                "gpu": "CUDA, FoldMatch default (fp32 weights, bf16 autocast inside ESM3.logits); P0.5: 4 x A100-SXM4-40GB, "
                       "torch 2.11.0+cu130 (enc_check also on an A100-PCIE-40GB: identical 576,399 / 580,608 bits)",
                "cpu": "fp32 CPU; P0.5: AMD EPYC 7763 Linux (16 ranks x 8 threads, torch 2.11.0+cu130 CPU path, MKL) "
                       "and Apple M1 Max macOS (7 threads, torch 2.11.0, Accelerate)",
                "mps": "Apple MPS fp32 (D18); P0.5: M1 Max, torch 2.11.0",
                "mixed": "query encodings from one device class, target codes from another (every ordered pair of the "
                         "device classes A4 runs)"},
        },
        "gates": [
            dict(id="A4-bits", metric="code_bits", applies_to=["gpu", "cpu", "mps"],
                 threshold={"gpu": A4_BITS, "cpu": A4_BITS, "mps": A4_BITS}, comparator=">=",
                 provisional={"gpu": 0.98, "cpu": 0.97, "mps": 0.97},
                 observed_P0_5=dict(pooled=pooled, per_set=per_set), decision="TIGHTENED (gpu 0.98 -> 0.990, cpu/mps 0.97 -> 0.990)",
                 reason=("(1) The provisional CPU value assumed CPU fp32 would sit further from the research bf16 codes "
                         "than the GPU path; P0.5 shows the opposite (CPU/MPS 0.99561, GPU 0.99511 pooled over 50.2M bits). "
                         "(2) At the provisional values the gate could not bind before the dlnE gate: dlnE on P_sub is "
                         f"driven by target-code flips (research q x device codes: median {m_rq[0]:.3f}-{m_rq[1]:.3f}, P95 "
                         f"{p_rq[0]:.3f}-{p_rq[1]:.3f}; device q x research codes: median {m_qr[0]:.3f}-{m_qr[1]:.3f}, "
                         f"P95 {p_qr[0]:.3f}-{p_qr[1]:.3f}), and its P95 grows roughly in proportion "
                         "to the bit disagreement, so the dlnE P95 bound 0.5 (~3x observed) corresponds to ~1.4% "
                         "disagreement; 0.990 (1.0%, 2.0-2.3x observed) keeps the two gates consistent. (3) Every "
                         "P0.5 set clears it: the lowest are the 50,000 A5 targets on GPU (25,000 of them reversed decoys; "
                         "0.99273 over 1.02e9 bits) and the 50-item enc_check (GPU 0.99275, CPU 0.99334). The residual "
                         "disagreement is mainly FoldMatch path vs research path (bf16-weight, batched SequenceOnlyESM3): "
                         "FoldMatch GPU vs FoldMatch CPU agree to 0.99739 and CPU Mac vs CPU Linux to 0.99997.")),
            dict(id="A4-dlnE", metric="dlnE", applies_to=["gpu", "cpu", "mps", "mixed"],
                 threshold=dict(median=A4_DLNE_MEDIAN, p95=A4_DLNE_P95), comparator="<=",
                 provisional=dict(median=0.1, p95=0.5), observed_P0_5=dl, decision="KEPT",
                 reason=(f"Observed over all {len(dd_keys)} device x device configs on P_sub ({man['hits']['n_pairs']} pairs): "
                         f"median {m_dd[0]:.3f}-{m_dd[1]:.3f}, P95 {p_dd[0]:.3f}-{p_dd[1]:.3f}, P99 <= {p99_dd[1]:.3f}, "
                         f"max <= {x_dd[1]:.3f}, i.e. 2x (median) and ~3x (P95) margin. The shipped "
                         "table's own split-vs-all stability is max |dlnE| 0.47 at E = 1 (ship_final.json), so a P95 of 0.5 "
                         "stays inside the calibration uncertainty; nothing in P0.5 argues for loosening, and tightening "
                         "would only shrink the margin against GPU-model / library drift without catching a failure "
                         "the A4-bits gate would miss.")),
            dict(id="A4-dF1", metric="dF1", applies_to=["gpu", "cpu", "mps", "mixed"], threshold=A4_DF1,
                 comparator="abs <=", provisional=0.01, observed_P0_5=cf, decision="KEPT (read as two-sided |dF1|)",
                 reason=(f"Observed |dF1| <= {dF1max:.5f} over {len(cf)} FoldMatch-path configs (pair-bootstrap 95% CIs "
                         f"within [{ci_lo:+.4f}, {ci_hi:+.4f}]); the research chain-set differences between SW settings it treated as "
                         "material were >= 0.03 (B/C vs A), and setting D vs A was +0.005 (no cost). 0.01 has >10x margin "
                         "and is not loosened; the provisional wording is read as two-sided because a shift of either "
                         "sign beyond 0.01 would indicate a pipeline change, not ESM3 noise.")),
            dict(id="A4-xcpu", metric="xcpu_bits", applies_to=["cpu"], threshold=A4_XCPU_BITS, comparator=">=",
                 provisional=0.995, observed_P0_5=dict(pooled=xd, per_item_min=xitem),
                 decision="TIGHTENED (0.995 -> 0.999)",
                 reason=("Observed CPU Mac (M1 Max, torch BLAS = Accelerate) vs CPU Linux (EPYC 7763, torch BLAS = MKL) "
                         "0.9999666 pooled over 50.2M "
                         "bits (1,678 bits differ), worst item 0.99955 over 2,598 items; fp16 rows cosine >= 0.99999989 "
                         "(rel. Frobenius 2.4e-5); MPS vs either CPU 0.99997. 0.995 would accept 150x the observed "
                         "cross-machine disagreement (e.g. a CPU path with reduced-precision matmuls); 0.999 is 30x the "
                         "pooled and 2.2x the worst per-item disagreement, so a correct fp32 CPU run on another machine "
                         "is not expected to fail it on a pooled fixture (F2c).")),
        ],
        "invariants": ["every embedding finite; max |per_residue_embedding| < 65,504 (P0.5 max 16,795 on the A5 targets)",
                       "0 <unk> tokens and token count = L + 2 for every P0.5 input (498 A5 targets carry X/U/B)",
                       "a Q90 / lnE computed from research encodings reproduces bench_q90_shipped.npy and the research "
                       "S_D matrices (P0.5: identical Q90; P_sub S max |d| 1e-4)"],
        "reported_not_gated": {
            "dQ90": {k: dict(median=v["median"], p95=v["p95"], max=v["max"]) for k, v in P["dQ90"].items()
                     if v and "median" in v and "research" not in k},
            "dQ90_largest_gpu_a5q": P.get("dQ90_top_gpu_a5q"),
            "q_cosine_median": {d: {s: v["median"] for s, v in P["q_cosine_vs_research"][d].items()} for d in devs},
            "full_matrix_dlnE_gpu_a5": {k: v for k, v in P["dlnE"]["full_a5"].get("gpu__gpu", {}).items()
                                        if k in ("research_hits", "union_hits", "hit_agreement")},
            "a5_preview_heldout_union_fp_ratio": {k: v["heldout_union"] for k, v in P.get("a5_preview", {}).items()},
        },
        "on_failure": "Stop and report (plan G8). The remedy is the v2 refit (plan P0.5): re-encode the panel and the "
                      "benchmark with FoldMatch, rerun ship.py / fit_final2.py unchanged, ship esm-codes-v2; v1 artefacts "
                      "are never edited. A threshold is never relaxed to make A4 pass.",
        "D18": ("MPS meets every CPU gate in P0.5 (bits 0.99561, dlnE median 0.051 / P95 0.159, dF1 +0.0004; MPS vs CPU "
                "0.99997) -> allow MPS. Condition: release the MPS allocator cache per protein (torch.mps.empty_cache); "
                "without it one process's footprint reached 83 GB after ~800 variable-length proteins (tsub + panel) "
                "on a 64 GB M1 Max, the machine swapped and throughput fell to ~40 res/s; with the per-protein release "
                "the driver memory stayed at 6.1-6.5 GiB and throughput was 700-900 res/s (Mac CPU fp32: 170-210). Mixed "
                "GPU/CPU/MPS use passes every mixed gate."),
    }
    jdump(out, a.out)
    log(f"wrote {a.out} sha256 {sha256(a.out)}")


def cmd_archive(a):
    """Durable compact copy of the P0.5 outputs (home file system): inputs, JSONs, logs, job scripts, chain-F1 results,
    small score files, and the encodings without the bulky parts (target-side q and most raw rows dropped; the A5 GPU
    encodings kept as query q + target codes, which is what A5 needs). Writes MD5SUMS."""
    import shutil
    dest = a.dest
    os.makedirs(dest, exist_ok=True)
    for f in ("esm3_parity.json", "FM_COMMIT"):
        shutil.copy2(f"{WORK}/{f}", f"{dest}/{f}")
    shutil.copy2(os.path.abspath(__file__), f"{dest}/esm3_parity.py")
    th = os.path.join(os.path.dirname(os.path.abspath(__file__)), "A4_THRESHOLDS.json")
    if os.path.exists(th):
        shutil.copy2(th, f"{dest}/A4_THRESHOLDS.json")
    for d in ("inputs", "logs", "chainf1", "jobs"):
        if os.path.isdir(f"{WORK}/{d}"):
            shutil.copytree(f"{WORK}/{d}", f"{dest}/{d}", dirs_exist_ok=True)
    os.makedirs(f"{dest}/scores", exist_ok=True)
    for f in glob.glob(f"{WORK}/scores/*.npz"):
        if not os.path.basename(f).startswith("full."):
            shutil.copy2(f, f"{dest}/scores/")
    os.makedirs(f"{dest}/enc", exist_ok=True)
    for f in sorted(glob.glob(f"{WORK}/enc/*.npz")):
        src, set_name = os.path.basename(f)[:-4].split(".")
        z = np.load(f)
        keep = ["ids", "keys", "lengths", "offsets", "codes", "maxabs", "min_abs_h", "unk", "meta_json"]
        if set_name in ("enccheck", "q200", "chains", "a5q"):
            keep.append("q")
        if set_name == "enccheck" and "raw" in z.files:
            keep.append("raw")
        with open(f"{dest}/enc/{src}.{set_name}.npz", "wb") as fh:
            np.savez(fh, **{k: z[k] for k in keep if k in z.files})
    lines = []
    for root, _, files in os.walk(dest):
        for fn in sorted(files):
            p = os.path.join(root, fn)
            if fn != "MD5SUMS":
                lines.append(f"{md5(p)}  {os.path.relpath(p, dest)}")
    open(f"{dest}/MD5SUMS", "w").write("\n".join(sorted(lines, key=lambda x: x[34:])) + "\n")
    log(f"archived {len(lines)} files to {dest}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("inputs")
    e = sub.add_parser("embed")
    e.add_argument("--device", required=True)
    e.add_argument("--devtag")
    e.add_argument("--sets", nargs="+", required=True)
    e.add_argument("--raw", nargs="*")
    e.add_argument("--rank", type=int, default=0)
    e.add_argument("--nranks", type=int, default=1)
    e.add_argument("--threads", type=int, default=0)
    e.add_argument("--flush", type=int, default=500)
    e.add_argument("--mps-empty-cache", action="store_true", help="torch.mps.empty_cache() after every protein")
    m = sub.add_parser("merge")
    m.add_argument("--devtag", required=True)
    m.add_argument("--sets", nargs="+", required=True)
    s = sub.add_parser("score")
    s.add_argument("--plan", nargs="+", required=True)
    s.add_argument("--nwork", type=int, default=int(os.environ.get("NWORK", "4")))
    c = sub.add_parser("chainf1")
    c.add_argument("--srcs", nargs="+", required=True)
    c.add_argument("--workers", type=int, default=int(os.environ.get("NWORK", "4")))
    sub.add_parser("analyze")
    t = sub.add_parser("thresholds")
    t.add_argument("--parity", default=f"{WORK}/esm3_parity.json")
    t.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "A4_THRESHOLDS.json"))
    r = sub.add_parser("archive")
    r.add_argument("--dest", default="/global/homes/j/jsegura/esm_codes_archive/p0/parity")
    a = ap.parse_args()
    {"inputs": cmd_inputs, "embed": cmd_embed, "merge": cmd_merge, "score": cmd_score, "chainf1": cmd_chainf1,
     "analyze": cmd_analyze, "thresholds": cmd_thresholds, "archive": cmd_archive}[a.cmd](a)


if __name__ == "__main__":
    main()
