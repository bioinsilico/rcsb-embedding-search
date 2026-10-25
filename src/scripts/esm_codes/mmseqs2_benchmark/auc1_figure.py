#!/usr/bin/env python
"""Figure of the full MMseqs2 benchmark (nbt.3988): per-query AUC1 against the fraction of queries.

The publication's Fig. 2a plots, for each tool, the fraction of queries (y) whose AUC up to the first false positive
is at least x (x). This figure has the axes swapped: x is the fraction of queries, y the AUC1 that this fraction of
queries reaches or exceeds (each method's per-query AUC1 sorted in descending order). The legend gives each method's
mean AUC1, the number the publication's Fig. 2b plots.

Input: the evaluator's output folders (auc1.py evaluate), <eval>/<method>/<protocol>/per_query.tsv.

    python auc1_figure.py --eval DIR --protocol paper --out fig_paper [--methods m1,m2,...]

Writes <out>.png, <out>.pdf and <out>.tsv (the curves, one column per method: the table view of the figure).
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np

# method folder -> (label, colour, line style, line width[, marker]); colours from the validated categorical palette
# (blue, orange, aqua pass all-pairs; BLAST in ink like the publication's black; FoldMatch's two scorings are two steps
# of one blue ramp, its two embedding thresholds solid (0.7) and dashed (0.8)). The refine variants of the Stage-1
# experiment (refine/r3_tables.py: fm-<esm|blosum>-t<07|08>-<V>) keep their scoring's colour and their threshold's
# line style and add sparse markers: V1 (exact re-scoring of the IVF-PQ top 10,000) round, V2 (re-scoring of the
# top 20,000) square, V3 (exact search) triangle; V0 is the benchmark itself (same style as foldmatch-*).
STYLE = {
    "foldmatch-esmcodes":     ("FoldMatch esm-codes, threshold 0.7", "#1c5cab", "-", 2.0),
    "foldmatch-esmcodes-t08": ("FoldMatch esm-codes, threshold 0.8", "#1c5cab", "--", 2.0),
    "foldmatch-blosum62":     ("FoldMatch BLOSUM62, threshold 0.7", "#6da7ec", "-", 2.0),
    "foldmatch-blosum62-t08": ("FoldMatch BLOSUM62, threshold 0.8", "#6da7ec", "--", 2.0),
    "mmseqs2-sen3":           ("MMseqs2 -s 10 (sen-3)", "#eb6834", "-", 1.6),
    "blast-k4000":            ("BLASTP (max 4,000 targets)", "#0b0b0b", "-", 1.6),
    "diamond":                ("DIAMOND ultra-sensitive", "#1baf7a", "-", 1.6),
}
REFINE = {"V0": ("", None), "V1": (", exact re-scoring", "o"), "V2": (", exact re-scoring of top 20,000", "s"),
          "V3": (", exact search", "^")}
for _sc, (_name, _col) in {"esm": ("esm-codes", "#1c5cab"), "blosum": ("BLOSUM62", "#6da7ec")}.items():
    for _t, (_tl, _ls) in {"t07": ("0.7", "-"), "t08": ("0.8", "--")}.items():
        for _v, (_suffix, _mk) in REFINE.items():
            STYLE[f"fm-{_sc}-{_t}-{_v}"] = (f"FoldMatch {_name}, threshold {_tl}{_suffix}", _col, _ls, 2.0, _mk)
DEFAULT_METHODS = ["foldmatch-esmcodes", "foldmatch-esmcodes-t08", "foldmatch-blosum62", "foldmatch-blosum62-t08",
                   "mmseqs2-sen3", "blast-k4000", "diamond"]
# the full-set runs (the paper's exact data, query.fasta x targetdb.fasta, sequences longer than 2,000 aa included;
# folders <method>-full, e.g. mmseqs2-sen3-full) keep their method's label and style (pass --title for the full set)
for _m in DEFAULT_METHODS:
    STYLE[f"{_m}-full"] = STYLE[_m]
PROTOCOL_NOTE = {
    "paper": "EvaluateResults as in the publication: first 4,000 hits, denominator min(family size, 4,000)",
    "spec": "whole hit list, denominator = all family members in the database",
    "spec_res4000": "first 4,000 targets, denominator capped at 4,000",
    "paper_noself": "as 'paper', without each query's own source sequence",
    "spec_noself": "as 'spec', without each query's own source sequence",
}


def per_query_auc1(path):
    out = []
    with open(path) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if r["evaluated"] == "1":
                out.append(float(r["auc1"]))
    return np.array(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--protocol", default="paper")
    ap.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default=None)
    ap.add_argument("--legend", choices=("auto", "inside", "outside"), default="auto",
                    help="legend inside the plot (upper right) or to its right; auto: outside for more than 7 curves")
    a = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = []
    for m in a.methods.split(","):
        p = os.path.join(a.eval, m, a.protocol, "per_query.tsv")
        if not os.path.exists(p):
            print(f"skipping {m}: no {p}")
            continue
        v = np.sort(per_query_auc1(p))[::-1]
        curves.append((m, v))
    if not curves:
        raise SystemExit("no method found")
    ns = {len(v) for _, v in curves}
    curves.sort(key=lambda mv: -mv[1].mean())

    ink, ink2, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    outside = a.legend == "outside" or (a.legend == "auto" and len(curves) > 7)
    fig, ax = plt.subplots(figsize=(12.0, 5.8) if outside else (7.2, 5.4), dpi=150)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    n_marked = 0
    for m, v in curves:
        label, color, ls, lw, *rest = STYLE.get(m, (m, "#52514e", "-", 1.6))
        marker = rest[0] if rest else None
        x = np.arange(1, len(v) + 1) / len(v)
        kw = {}
        if marker:
            # sparse markers (one per 0.1 of the fraction axis, staggered between curves), ringed in the surface colour
            off = 0.05 + 0.025 * (n_marked % 4)
            idx = np.unique(np.clip(np.round((np.arange(off, 1.0, 0.1)) * len(v)).astype(int), 1, len(v)))
            kw = dict(marker=marker, markevery=list(idx), markersize=5.5, markerfacecolor=color,
                      markeredgecolor=surface, markeredgewidth=1.0)
            n_marked += 1
        ax.plot(np.concatenate([[0.0], x]), np.concatenate([[v[0]], v]), color=color, ls=ls, lw=lw,
                label=f"{label}  ({v.mean():.3f})", solid_capstyle="round", drawstyle="steps-pre", **kw)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Fraction of queries", color=ink)
    ax.set_ylabel("AUC up to the first false positive", color=ink)
    ax.grid(True, color=grid, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(ink2)
    ax.tick_params(colors=ink2)
    if outside:
        leg = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=8.5,
                        title="mean AUC1 in brackets", title_fontsize=8, handlelength=3.2)
    else:
        leg = ax.legend(loc="upper right", frameon=False, fontsize=8.5, title="mean AUC1 in brackets",
                        title_fontsize=8)
    for t in leg.get_texts():
        t.set_color(ink)
    nq = f"{sorted(ns)[0]:,}" if len(ns) == 1 else "/".join(f"{n:,}" for n in sorted(ns))
    title = a.title or f"MMseqs2 benchmark, {nq} queries (<= 2,000 aa); AUC1 protocol '{a.protocol}'"
    ax.set_title(title, fontsize=10, color=ink, loc="left")
    note = PROTOCOL_NOTE.get(a.protocol)
    if note:
        fig.text(0.01, 0.005, note, fontsize=7, color=ink2)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(a.out + ".png")
    fig.savefig(a.out + ".pdf")

    n = max(len(v) for _, v in curves)
    with open(a.out + ".tsv", "w") as fh:
        fh.write("fraction_of_queries\t" + "\t".join(m for m, _ in curves) + "\n")
        for i in range(n):
            vals = [f"{v[i]:.6f}" if i < len(v) else "" for _, v in curves]
            fh.write(f"{(i + 1) / n:.6f}\t" + "\t".join(vals) + "\n")
    for m, v in curves:
        print(f"{m:24s} n {len(v)}  mean AUC1 {v.mean():.4f}  median {np.median(v):.4f}  "
              f"AUC1=1 {np.mean(v == 1):.3f}  AUC1=0 {np.mean(v == 0):.3f}")


if __name__ == "__main__":
    main()
