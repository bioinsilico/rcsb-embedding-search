#!/usr/bin/env python
"""fm_cuda_probe.py -- which precision would a similarity GEMM get inside FoldMatch's own CUDA context?

BLAS-divergence study (FoldMatch esm-codes-v1, plan decision D2), A100 part.  Runs on a GPU node in the 'foldmatch'
env (the installed foldmatch package).  It records the torch TF32 / fp32-matmul-precision / autocast state and computes
sim = qp . decode(codes)^T with a plain torch.matmul (fp32 inputs, the way a FoldMatch implementation would write it)
at these points of ONE process:

  fresh            right after `import torch` (nothing else imported)
  after_import     after importing FoldMatch the way its CLI / inference entry points do (foldmatch, esm, lightning)
  trainer_step     inside LightningModule.predict_step, Trainer configured like FoldMatch's inference entry points
                   (num_nodes, accelerator, devices, strategy, logger=False; no precision -> '32-true')
  esm_inside       INSIDE FoldMatch's ESM3 forward (EsmModule path: esm3.forward_and_sample(esm3.encode(prot),
                   SamplingConfig(return_per_residue_embeddings=True))) via a forward hook on esm3.transformer, i.e.
                   within ESM3.logits' torch.autocast(bfloat16) scope
  esm_after        in the same predict_step right after the ESM3 call returned
  after_trainer    in the main process after trainer.predict returned

Every sim is compared bit for bit with an explicit FP32 GEMM (allow_tf32 False, 'highest'; = blas_div torch_cuda),
an explicit TF32 GEMM (= blas_div torch_cuda_tf32) and the fixed-order C reference (spike sim_codes), and scored with
the fixed-order spike kernel (S_f, S_r) to show the score effect.  Also scans the installed foldmatch / esm / lightning
sources for code that changes the global matmul precision or opens autocast scopes.

usage: python fm_cuda_probe.py --out OUT.json [--no-esm] [--device cuda:0]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import blas_div as BD  # noqa: E402  (sets the BLAS thread variables before numpy)
import numpy as np  # noqa: E402

T0 = time.time()
SEQ = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"  # ubiquitin, 76 aa
GLOBAL_RE = re.compile(r"set_float32_matmul_precision\(|\.allow_tf32\s*=|fp32_precision\s*=|"
                       r"TORCH_ALLOW_TF32_CUBLAS_OVERRIDE|NVIDIA_TF32_OVERRIDE|allow_bf16_reduced_precision_reduction\s*=|"
                       r"allow_fp16_reduced_precision_reduction\s*=")
SCOPED_RE = re.compile(r"autocast\(|\.half\(\)|bfloat16|precision\s*=")


def log(*a):
    print(f"[{time.time() - T0:7.1f}s] " + " ".join(str(x) for x in a), flush=True)


def flags(torch):
    d = {}
    for nm, fn in (("matmul_allow_tf32", lambda: bool(torch.backends.cuda.matmul.allow_tf32)),
                   ("cudnn_allow_tf32", lambda: bool(torch.backends.cudnn.allow_tf32)),
                   ("float32_matmul_precision", torch.get_float32_matmul_precision),
                   ("matmul_allow_fp16_reduced", lambda: bool(
                       torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)),
                   ("matmul_allow_bf16_reduced", lambda: bool(
                       torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)),
                   ("fp32_precision_global", lambda: torch.backends.fp32_precision),
                   ("fp32_precision_cuda_matmul", lambda: torch.backends.cuda.matmul.fp32_precision),
                   ("fp32_precision_cudnn", lambda: torch.backends.cudnn.fp32_precision),
                   ("autocast_cuda_enabled", lambda: bool(torch.is_autocast_enabled("cuda"))),
                   ("autocast_cuda_dtype", lambda: str(torch.get_autocast_dtype("cuda"))),
                   ("inference_mode", lambda: bool(torch.is_inference_mode_enabled())),
                   ("grad_enabled", lambda: bool(torch.is_grad_enabled())),
                   ("default_dtype", lambda: str(torch.get_default_dtype()))):
        try:
            d[nm] = fn()
        except Exception as e:  # noqa: BLE001
            d[nm] = f"n/a: {type(e).__name__}: {e}"[:200]
    return d


class Payload:
    """One real (query, target block) per shape: production block (2^17 sim floats) and research chunk (2^20)."""

    def __init__(self, data):
        B = BD.load_bench(data)
        rows = BD.query_rows("q150", data)
        lens = B["qlen"][rows]
        r = int(rows[np.argsort(lens, kind="stable")[len(lens) // 2]])   # median-length q150 member
        q16 = np.asarray(B["q"][B["qoff"][r]:B["qoff"][r + 1]])
        SL, F = BD.spike()
        self.F = F
        self.qp = F.query_prep(q16)
        n = len(self.qp)
        self.cases = []
        for name, rule, bf in (("prod_2^17", "prod", 1 << 17), ("research_2^20", "research", 1 << 20)):
            off = B["toff"]
            blk = BD.make_blocks(dict(block_rule=rule, block_floats=bf), off, n)[0]
            k0, k1 = blk
            j0, j1 = int(off[k0]), int(off[k1])
            codes = np.asarray(B["codes"][j0:j1])
            col = (off[k0:k1 + 1] - j0).astype(np.int64)
            ref = F.sim_codes(self.qp, codes)
            sf, sr = F.score_fwd_rev(ref, col)
            self.cases.append(dict(name=name, row=r, n=n, k0=int(k0), k1=int(k1), m=j1 - j0, codes=codes, col=col,
                                   D=BD.decode(codes), ref=ref, sf=sf, sr=sr,
                                   s64=BD.oracle_sim(q16, codes)))
        log(f"payload: q150 row {r} (n={n}); " + ", ".join(f"{c['name']}: targets {c['k0']}..{c['k1']} m={c['m']}"
                                                        for c in self.cases))


def gemm(torch, dev, P, case):
    qt = torch.from_numpy(np.ascontiguousarray(P.qp, np.float32)).to(dev)
    Dt = torch.from_numpy(case["D"]).to(dev)
    S = torch.matmul(qt, Dt.T)
    return str(S.dtype), np.ascontiguousarray(S.float().cpu().numpy(), np.float32)


def explicit(torch, dev, P, case, tf32):
    old = (torch.backends.cuda.matmul.allow_tf32, torch.get_float32_matmul_precision())
    try:
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.set_float32_matmul_precision("high" if tf32 else "highest")
        return gemm(torch, dev, P, case)[1]
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old[0]
        torch.set_float32_matmul_precision(old[1])


def summarize(P, case, dtype, sim, fp32, tf32):
    F = P.F
    sf, sr = F.score_fwd_rev(np.ascontiguousarray(sim), case["col"])
    d = dict(result_dtype=dtype, cells=int(sim.size),
             bit_equal_explicit_fp32=float(BD.bits_equal(sim, fp32).mean()),
             bit_equal_explicit_tf32=float(BD.bits_equal(sim, tf32).mean()),
             bit_equal_fixed_c=float(BD.bits_equal(sim, case["ref"]).mean()),
             max_abs_vs_fixed_c=float(np.abs(sim.astype(np.float64) - case["ref"]).max()),
             max_abs_vs_float64=float(np.abs(sim.astype(np.float64).ravel() - case["s64"].ravel()).max()),
             S_f_bit_equal_fixed_c=float(BD.bits_equal(sf, case["sf"]).mean()),
             S_r_bit_equal_fixed_c=float(BD.bits_equal(sr, case["sr"]).mean()),
             S_f_max_abs_vs_fixed_c=float(np.abs(sf.astype(np.float64) - case["sf"]).max()),
             S_r_max_abs_vs_fixed_c=float(np.abs(sr.astype(np.float64) - case["sr"]).max()),
             S_max_abs_vs_fixed_c=float(np.abs((sf.astype(np.float64) - sr) - (case["sf"].astype(np.float64)
                                                                                - case["sr"])).max()),
             S_f_fixed_c=case["sf"].tolist()[:8], S_f_this=sf.tolist()[:8])
    d["equals"] = ("explicit_fp32" if d["bit_equal_explicit_fp32"] == 1.0 else
                   "explicit_tf32" if d["bit_equal_explicit_tf32"] == 1.0 else "neither")
    return d


def point(torch, dev, P, name, res):
    """flags + sim at this point for every payload case (explicit references recomputed here, same device)."""
    rec = dict(flags=flags(torch), cases={})
    for case in P.cases:
        dtype, sim = gemm(torch, dev, P, case)
        fp32 = explicit(torch, dev, P, case, False)
        tf32 = explicit(torch, dev, P, case, True)
        rec["cases"][case["name"]] = summarize(P, case, dtype, sim, fp32, tf32)
    rec["flags_after_explicit_restore"] = flags(torch)
    res["points"][name] = rec
    log(name, json.dumps({k: rec["flags"][k] for k in ("matmul_allow_tf32", "float32_matmul_precision",
                                                         "fp32_precision_cuda_matmul", "autocast_cuda_enabled")}),
        {c: (v["result_dtype"], v["equals"], f"{v['max_abs_vs_fixed_c']:.3g}", f"{v['S_max_abs_vs_fixed_c']:.3g}")
         for c, v in rec["cases"].items()})


def point_inside(torch, dev, P, name, res):
    """inside an autocast scope the explicit-reference helpers would also be autocast: compute only the plain GEMM here
    and compare with the references computed outside (stored in res['ref_outside'])."""
    rec = dict(flags=flags(torch), cases={})
    for case in P.cases:
        dtype, sim = gemm(torch, dev, P, case)
        rec["cases"][case["name"]] = dict(dtype=dtype, sim=sim)
    res.setdefault("_inside", {})[name] = rec


def scan_sources():
    import importlib.util
    out = {}
    for pkg, full in (("foldmatch", True), ("esm", True), ("lightning", False), ("lightning_fabric", False),
                      ("pytorch_lightning", False), ("faiss", False)):
        spec = importlib.util.find_spec(pkg)
        if spec is None or not spec.submodule_search_locations:
            out[pkg] = dict(found=False)
            continue
        root = list(spec.submodule_search_locations)[0]
        glob_hits, scoped = [], []
        for dp, _, fns in os.walk(root):
            for fn in fns:
                if not fn.endswith(".py"):
                    continue
                p = os.path.join(dp, fn)
                try:
                    lines = open(p, encoding="utf-8", errors="replace").read().splitlines()
                except OSError:
                    continue
                for i, ln in enumerate(lines, 1):
                    if GLOBAL_RE.search(ln):
                        glob_hits.append(f"{os.path.relpath(p, root)}:{i}: {ln.strip()[:160]}")
                    elif full and SCOPED_RE.search(ln):
                        scoped.append(f"{os.path.relpath(p, root)}:{i}: {ln.strip()[:160]}")
        ver = None
        try:
            from importlib.metadata import version
            ver = version({"lightning_fabric": "lightning", "pytorch_lightning": "pytorch_lightning",
                           "faiss": "faiss-cpu"}.get(pkg, pkg))
        except Exception:  # noqa: BLE001
            pass
        out[pkg] = dict(found=True, root=root, version=ver, global_state_lines=glob_hits,
                        scoped_lines=scoped if full else "not scanned (global-state patterns only)")
    try:
        from importlib.metadata import distribution
        out["foldmatch"]["direct_url"] = distribution("foldmatch").read_text("direct_url.json")
    except Exception:  # noqa: BLE001
        pass
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=BD.DATA_DEFAULT)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-esm", action="store_true")
    a = ap.parse_args()
    res = dict(what="FoldMatch CUDA-context precision probe (blas_div, plan D2)", created=time.strftime("%F %T"),
               argv=sys.argv, host=BD.host_info(),
               env={k: os.environ.get(k) for k in ("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "NVIDIA_TF32_OVERRIDE",
                                                   "CUBLAS_WORKSPACE_CONFIG", "CUDA_VISIBLE_DEVICES",
                                                   "HF_HUB_OFFLINE", "SLURM_JOB_ID")},
               points={}, errors={})
    import torch
    res["torch_fresh_flags"] = flags(torch)
    dev = torch.device(a.device)
    torch.cuda.set_device(dev)
    res["device"] = dict(name=torch.cuda.get_device_name(dev), capability=list(torch.cuda.get_device_capability(dev)),
                         torch=torch.__version__, cuda=torch.version.cuda,
                         cublas=getattr(torch._C, "_cuda_getCompiledVersion", lambda: None)())
    P = Payload(a.data)
    res["payload"] = [dict({k: v for k, v in c.items() if k in ("name", "row", "n", "k0", "k1", "m")},
                           fixed_c_S_f=c["sf"].tolist(), fixed_c_S_r=c["sr"].tolist()) for c in P.cases]
    point(torch, dev, P, "fresh", res)

    # --- FoldMatch imports (what its CLI / inference entry points import) ---
    imp = {}
    for mod in ("foldmatch", "foldmatch.foldmatch", "foldmatch.utils.model", "foldmatch.modules.esm_module",
                "foldmatch.modules.chain_complete_module", "foldmatch.inference.esm_inference",
                "foldmatch.inference.chain_complete_inference", "foldmatch.search.alignment",
                "foldmatch.search.scoring_system", "foldmatch.search.embedding_computer", "foldmatch.cli.search",
                "foldmatch.cli.inference"):
        try:
            __import__(mod)
            imp[mod] = "ok"
        except Exception as e:  # noqa: BLE001
            imp[mod] = f"{type(e).__name__}: {e}"[:300]
    res["foldmatch_imports"] = imp
    point(torch, dev, P, "after_import", res)

    # --- Lightning Trainer like FoldMatch's inference entry points ---
    from lightning import Trainer
    from torch.utils.data import DataLoader
    try:
        from foldmatch.modules.esm_module import EsmModule
    except Exception as e:  # noqa: BLE001
        EsmModule = None
        res["errors"]["EsmModule_import"] = str(e)
    from lightning import LightningModule

    class PlainProbe(LightningModule):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(1, 1)          # a parameter so Lightning has something to move

        def predict_step(self, batch, batch_idx):
            point(torch, self.device, P, "trainer_step", res)
            return 0

    def trainer():
        # FoldMatch: Trainer(callbacks=..., num_nodes=num_nodes, accelerator=accelerator, devices=devices,
        #                    strategy=strategy, logger=False)   (no precision argument)
        return Trainer(callbacks=None, num_nodes=1, accelerator="gpu", devices=[dev.index or 0], strategy="auto",
                       logger=False)

    try:
        t1 = trainer()
        t1.predict(PlainProbe(), DataLoader([0], batch_size=1, collate_fn=lambda x: x), return_predictions=True)
        res["trainer_plain"] = dict(precision=str(getattr(t1, "precision", None)),
                                    precision_plugin=type(t1.precision_plugin).__name__,
                                    strategy=type(t1.strategy).__name__,
                                    cluster_environment=type(getattr(t1.strategy, "cluster_environment", None)).__name__)
    except Exception:  # noqa: BLE001
        res["errors"]["trainer_plain"] = traceback.format_exc()[-3000:]
        log("trainer_plain failed:", res["errors"]["trainer_plain"][-500:])

    # --- FoldMatch's ESM3 forward (EsmModule path) inside the same kind of Trainer ---
    if not a.no_esm and EsmModule is not None:
        try:
            from esm.sdk.api import ESMProtein
            from foldmatch.utils.model import get_residue_model
            t = time.time()
            esm3 = get_residue_model(dev)
            res["esm3_load_s"] = time.time() - t
            res["esm3_param_dtype"] = str(next(esm3.parameters()).dtype)
            log(f"ESM3 loaded in {res['esm3_load_s']:.0f}s, params {res['esm3_param_dtype']}")
            emb_info = {}

            class EsmProbe(EsmModule):
                def predict_step(self, prot_batch, batch_idx):
                    h = self.esm3.transformer.register_forward_hook(
                        lambda mod, inp, out: point_inside(torch, self.device, P, "esm_inside", res))
                    try:
                        embs, names = super().predict_step(prot_batch, batch_idx)   # FoldMatch's own code path
                    finally:
                        h.remove()
                    emb_info.update(dtype=str(embs[0].dtype), shape=list(embs[0].shape), names=list(names))
                    point(torch, self.device, P, "esm_after", res)
                    return 0

            tr = trainer()
            tr.predict(EsmProbe(esm3), DataLoader([(ESMProtein(sequence=SEQ), "ubiquitin")], batch_size=1,
                                                  collate_fn=lambda x: x), return_predictions=True)
            res["esm_embedding"] = emb_info
            res["trainer_esm"] = dict(precision=str(getattr(tr, "precision", None)),
                                      precision_plugin=type(tr.precision_plugin).__name__)
        except Exception:  # noqa: BLE001
            res["errors"]["esm"] = traceback.format_exc()[-3000:]
            log("esm part failed:", res["errors"]["esm"][-800:])
    # inside-scope GEMMs: compare against references computed outside the autocast scope
    for name, rec in res.pop("_inside", {}).items():
        out = dict(flags=rec["flags"], cases={})
        for case in P.cases:
            x = rec["cases"][case["name"]]
            fp32 = explicit(torch, dev, P, case, False)
            tf32 = explicit(torch, dev, P, case, True)
            out["cases"][case["name"]] = summarize(P, case, x["dtype"], x["sim"], fp32, tf32)
        res["points"][name] = out
        log(name, json.dumps({k: out["flags"][k] for k in ("autocast_cuda_enabled", "autocast_cuda_dtype")}),
            {c: (v["result_dtype"], v["equals"], f"{v['max_abs_vs_fixed_c']:.3g}", f"{v['S_max_abs_vs_fixed_c']:.3g}")
             for c, v in out["cases"].items()})
    point(torch, dev, P, "after_trainer", res)
    res["sources"] = scan_sources()
    res["torch_info"] = BD.torch_info(torch)
    res["summary"] = {k: dict(flags={f: v["flags"].get(f) for f in ("matmul_allow_tf32", "float32_matmul_precision",
                                                                     "fp32_precision_cuda_matmul",
                                                                     "autocast_cuda_enabled", "autocast_cuda_dtype")},
                              cases={c: dict(result_dtype=x["result_dtype"], equals=x["equals"],
                                             max_abs_vs_fixed_c=x["max_abs_vs_fixed_c"],
                                             S_max_abs_vs_fixed_c=x["S_max_abs_vs_fixed_c"],
                                             S_f_bit_equal_fixed_c=x["S_f_bit_equal_fixed_c"])
                                     for c, x in v["cases"].items()})
                      for k, v in res["points"].items()}
    res["elapsed_s"] = time.time() - T0
    BD.jdump(res, a.out)
    print(json.dumps(res["summary"], indent=1, default=BD._jdefault))
    log("->", a.out)


if __name__ == "__main__":
    main()
