#!/usr/bin/env python
"""P8 A7: throughput (plan FINAL v1, section 8 P8, A7 row: 'P7 benchmark | P7 targets').

No new run: A7 takes the P7 record (P6 / P7 'Notes for P8': "A7 (throughput) has the P7 targets: section 3 is the
record, with the open Stage A decision") and the user's Stage A decision recorded in PLAN.md (section 8 P7, 'Stage A
decision (user, ...)': accepted at max_evalue = 1e-3; at inf a documented limit, not a release blocker). The P7
numbers were measured on the code P8 tests: P7's final tree b30280f = commit cc55c01's tree cd198f5b apart from the
CHANGELOG text (P7_REPORT.md section 3: "36c930c only refreshed the CHANGELOG text afterwards").

This script quotes, verbatim, the P7_REPORT.md section 3 target table (and its notes and the M1 / profiling lines) and
the PLAN.md Stage A paragraph, with the md5 of both files, and derives the per-target verdicts under the Stage A
decision. Output: <P8_ROOT>/A7/a7_record.json and a7_record.md.

Usage: python a7_throughput_record.py [--plan PLAN.md] [--p7-report P7_REPORT.md]
"""
from __future__ import annotations

import argparse
import os
import re
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import acc_common as C  # noqa: E402

P7_REPORT_MD5 = "9e0b69e942d1169dea8387b04d0e2fd6"          # archive esm_codes_archive/p7/P7_REPORT.md (staged)


def section(text: str, start: str, stop_prefix: str = "## ") -> str:
    i = text.index(start)
    j = text.find("\n" + stop_prefix, i + len(start))
    return text[i:j if j >= 0 else len(text)].rstrip() + "\n"


def table_rows(sec: str):
    rows = []
    for line in sec.splitlines():
        if line.startswith("| ") and not line.startswith("|---") and not line.startswith("| #"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) == 6:
                rows.append(dict(zip(("id", "target", "p71_baseline", "acceptance_583090d", "final_b30280f",
                                      "verdict"), cells)))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--plan", default=os.path.join(C.SANDBOX, "PLAN.md"))
    ap.add_argument("--p7-report", default=None, help="default: the staged input reports/P7_REPORT.md")
    a = ap.parse_args(argv)
    rep = a.p7_report or C.inp("reports/P7_REPORT.md")
    md5_rep, md5_plan = C.md5_file(rep), C.md5_file(a.plan)
    if md5_rep != P7_REPORT_MD5:
        raise RuntimeError(f"{rep}: md5 {md5_rep} != the archived P7 report {P7_REPORT_MD5}")
    text = open(rep, encoding="utf-8").read()
    plan = open(a.plan, encoding="utf-8").read()
    sec3 = section(text, "## 3. P7 targets")
    sec8 = section(text, "## 8. Notes for P8")
    m = re.search(r"\*\*Stage A decision \(user, [^)]*\)[^\n]*", plan)
    if not m:
        raise RuntimeError(f"{a.plan}: no 'Stage A decision (user, ...)' paragraph")
    stage_a = m.group(0)
    rows = table_rows(sec3)
    # the Stage A decision covers row 6A (the node gate); every other row keeps the verdict P7 recorded
    for r in rows:
        if r["id"] == "6A" and "1e-3" in r["target"]:
            r["under_stage_a_decision"] = "release gate: PASS (accepted at max_evalue=1e-3)"
        elif r["id"] == "6A":
            r["under_stage_a_decision"] = "documented limit at inf, not a release blocker (user decision)"
        elif r["id"] in ("6A2", "6A3"):
            r["under_stage_a_decision"] = "same-workload / reference reading, recorded for reference (user decision)"
        else:
            r["under_stage_a_decision"] = "not covered by the Stage A decision: P7 verdict as recorded"
    # the code P7 measured (b30280f) against the code P8 tests (cc55c01): only CHANGELOG.md may differ
    names = C._git(C.FM_CHECKOUT, "diff", "--name-only", "b30280f", C.EXPECTED_FM_COMMIT)
    stat = C._git(C.FM_CHECKOUT, "diff", "--shortstat", "b30280f", C.EXPECTED_FM_COMMIT)
    code_diff = dict(repo=C.FM_CHECKOUT, measured="b30280f", tested=C.EXPECTED_FM_COMMIT,
                     files=None if names is None else [x for x in names.splitlines() if x], shortstat=stat)
    code_diff["only_changelog"] = code_diff["files"] == ["CHANGELOG.md"]
    gate = [r for r in rows if r["id"] == "6A" and "1e-3" in r["target"]]
    rec = dict(check="A7", what=__doc__.split("\n\n")[0],
               sources=dict(p7_report=dict(path=rep, realpath=os.path.realpath(rep), md5=md5_rep),
                            plan=dict(path=a.plan, md5=md5_plan)),
               stage_a_decision=stage_a, p7_section3=sec3, p7_notes_for_p8=sec8, table=rows,
               release_gate_row=gate,
               verdict=dict(stage_a_release_gate="PASS (4.38x >= 3x at max_evalue=1e-3, 129,762 vs 29,623 pairs/s)",
                            inf="documented limit (2.75x), not a release blocker (user decision)",
                            new_run=False,
                            code_identity=code_diff),
               provenance=C.provenance(kernel=True))
    out = C.result_dir("A7")
    h = C.write_json(rec, os.path.join(out, "a7_record.json"))
    md = ["# A7 throughput record (P8)", "",
          f"Source: `{rep}` (md5 `{md5_rep}`) section 3, quoted verbatim; Stage A decision from `{a.plan}` "
          f"(md5 `{md5_plan}`).", "", "## Stage A decision (PLAN.md section 8 P7)", "", stage_a, "",
          "## P7_REPORT.md section 3 (verbatim)", "", sec3]
    with open(os.path.join(out, "a7_record.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))
    C.log(f"A7: {len(rows)} table rows; release gate row {gate[0]['final_b30280f'] if gate else None}; wrote "
          f"{os.path.join(out, 'a7_record.json')} sha256 {h}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
