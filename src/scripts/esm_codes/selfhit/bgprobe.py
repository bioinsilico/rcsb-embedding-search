#!/usr/bin/env python
"""Self-hit study, step 5: is the high row background mu_i of weak self hits specific to the protein's own residues
(its residue codes all alike) or general (its query encodings resemble every protein's codes)? numpy only.

The row mean of a pair's similarity is linear in the codes: mu_i = qp_i . mean_j(d_j). So, closed form, per query
residue i:
  * own background     mu_i^self = qp_i . dbar_self                        (the self matrix row mean, S7)
  * panel background   mu_i^P    = qp_i . dbar_P,  sigma_i^P = sqrt(qp_i' C_P qp_i - (mu_i^P)^2)
                       over all 355,626 panel residues (C_P = D_P' D_P / M_P)
  * code homogeneity   h = |dbar_self| / 8 (1 = every residue has the same code; the panel's own is |dbar_P| / 8)
  * z_ii^P = (s_ii - mu_i^P) / sigma_i^P: the diagonal standardised against the fixed panel background instead of
    the pair's own rows (the row half of candidate C1 in v2probe.py).

    PYTHONPATH=<wt-P6/src> python bgprobe.py DATA OUT.npz
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

from foldmatch.search import scoring_system as ss
from foldmatch.search import swkern

SETS = ("u50_q200", "u50_db800", "scop_q", "pdb_r600")


def main():
    data, out = sys.argv[1:3]
    spec = ss.prepare(ss.get("esm-codes-v1"), local_files_only=True).for_worker()
    pc, po = spec.load_panel()
    DP = swkern.decode_codes(np.ascontiguousarray(pc)).astype(np.float64)
    dbar_p = DP.mean(0)
    C_p = DP.T @ DP / len(DP)
    res = {}
    for name in SETS:
        z = np.load(os.path.join(data, "enc", f"{name}.npz"), allow_pickle=False)
        off = z["offsets"]
        cols = {k: [] for k in ("mu_self", "mu_p", "sd_p", "h", "zP_ii", "s_ii", "zP_pos")}
        for i in range(len(off) - 1):
            qp = swkern.query_prep(np.ascontiguousarray(z["q"][off[i]:off[i + 1]])).astype(np.float64)
            D = swkern.decode_codes(np.ascontiguousarray(z["codes"][off[i]:off[i + 1]])).astype(np.float64)
            dbar = D.mean(0)
            s_ii = (qp * D).sum(1)
            mu_p = qp @ dbar_p
            sd_p = np.sqrt(np.maximum(np.einsum("ik,kl,il->i", qp, C_p, qp) - mu_p ** 2, 0.0))
            zp = (s_ii - mu_p) / (sd_p + 1e-8)
            cols["mu_self"].append(float((qp @ dbar).mean()))
            cols["mu_p"].append(float(mu_p.mean()))
            cols["sd_p"].append(float(sd_p.mean()))
            cols["h"].append(float(np.linalg.norm(dbar) / 8))
            cols["zP_ii"].append(float(zp.mean()))
            cols["s_ii"].append(float(s_ii.mean()))
            cols["zP_pos"].append(float((zp > 1.0).mean()))
        for k, v in cols.items():
            res[f"{name}__{k}"] = np.array(v)
        res[f"{name}__ids"] = z["ids"]
    res["panel_h"] = np.array(np.linalg.norm(dbar_p) / 8)
    res["meta_json"] = json.dumps(dict(sets=SETS, panel_residues=int(len(DP))))
    np.savez(out, **res)
    print("panel |dbar|/8 =", float(res["panel_h"]))
    for name in SETS:
        print(name, {k: round(float(np.median(res[f"{name}__{k}"])), 3)
                     for k in ("mu_self", "mu_p", "sd_p", "h", "zP_ii", "s_ii")})


if __name__ == "__main__":
    main()
