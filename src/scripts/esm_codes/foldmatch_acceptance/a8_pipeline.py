#!/usr/bin/env python
"""P8 A8 (plan FINAL v1, section 8 P8): run the real ``fm-search`` CLI in this process
(``foldmatch.cli.search.main``, what the console script calls) and save, beside its own output, what its esm-codes
query run did inside - for A8's analysis and as A5's production query encodings.

Usage:  python a8_pipeline.py --capture-dir DIR -- query sequences --db-path ... --scoring esm-codes ...

The wrappers below pass every argument and result through unchanged (they only copy):
  * ``EmbeddingComputer.in_input_order``: after the original, the run's query chain embeddings read back from its own
    shards (``get_embedding_batches()``) -> ``query_embeddings.npz`` (ids, float32 [n, d]);
  * ``foldmatch.cli.search._filter_results_by_threshold``: the raw Stage-1 (FAISS) results before the threshold, and
    the threshold -> ``stage1_raw.npz`` (+ counts before / after in ``run.json``);
  * ``foldmatch.search.code_alignment.align_candidates_esm_codes``: a copy of the run's query-encoding memmap before
    the CLI deletes it (``queries.f16``, ``query_encodings.json``: entries, rows, width, system, sha256), the query
    sequences, N and the options of the call; the Stage-2 ``stats`` (a dict passed in when the CLI passes none: the
    per-query Q90, workers, pairs, survivors); the returned hits at full precision (``hits.npz``, fm_stage2
    .hits_arrays); and the call's wall time;
  * the CLI's console log is the job's stdout.
``run.json`` also carries the provenance record (acc_common.provenance: FoldMatch commit / file / install check,
kernel info, host, devices, BLAS) and the exit status.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

T0 = time.perf_counter()
import acc_common as AC  # noqa: E402  (first: BLAS threads = 1)
import numpy as np  # noqa: E402


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        raise SystemExit("usage: a8_pipeline.py --capture-dir DIR [--label L] -- <fm-search args>")
    k = argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-dir", required=True)
    ap.add_argument("--label", default="")
    a = ap.parse_args(argv[:k])
    fm_args = argv[k + 1:]
    out = a.capture_dir
    os.makedirs(out, exist_ok=True)

    import foldmatch.cli.search as cli
    import foldmatch.search.code_alignment as ca
    import foldmatch.search.output as output
    from foldmatch.search.embedding_computer import EmbeddingComputer
    import fm_stage2 as F

    rec = dict(what="P8 A8: fm-search run in-process with pass-through capture wrappers (a8_pipeline.py)",
               label=a.label, fm_search_argv=fm_args, phases={}, stage2_calls=0, import_s=time.perf_counter() - T0)
    orig_order, orig_filter = EmbeddingComputer.in_input_order, cli._filter_results_by_threshold
    orig_s2, orig_out = ca.align_candidates_esm_codes, output.write_aligned_results

    def in_input_order(self, results):
        res = orig_order(self, results)
        t0 = time.perf_counter()
        ids, embs = [], []
        for ib, eb in self.get_embedding_batches(rows=False):
            ids.extend(ib)
            embs.append(np.asarray(eb, np.float32))
        E = np.concatenate(embs, 0) if embs else np.zeros((0, 0), np.float32)
        rec["query_embeddings_sha256"] = F.save_npz(os.path.join(out, "query_embeddings.npz"),
                                                    ids=np.array(ids), emb=E)
        rec["query_embeddings"] = dict(n=len(ids), d=int(E.shape[1]) if E.size else 0,
                                       capture_s=time.perf_counter() - t0)
        return res

    def filter_by_threshold(results, threshold):
        t0 = time.perf_counter()
        qids = list(results)
        counts = np.array([len(results[q][0]) for q in qids], np.int64)
        tids = np.array([t for q in qids for t in results[q][0]])
        scores = np.array([float(s) for q in qids for s in results[q][1]], np.float32)
        rec["stage1_raw_sha256"] = F.save_npz(os.path.join(out, "stage1_raw.npz"), qids=np.array(qids),
                                              counts=counts, tids=tids, scores=scores)
        res = orig_filter(results, threshold)
        rec["stage1"] = dict(threshold=threshold, queries=len(qids), candidates_raw=int(counts.sum()),
                             candidates_after_threshold=int(sum(len(v[0]) for v in res.values())),
                             max_per_query_raw=int(counts.max()) if counts.size else 0,
                             capture_s=time.perf_counter() - t0)
        return res

    def stage2(query_index, prefilter_results, code_store, fetch, prepared, db_residues, pool, **kw):
        rec["stage2_calls"] += 1
        t0 = time.perf_counter()
        shutil.copyfile(query_index.path, os.path.join(out, "queries.f16"))
        enc = dict(rows=int(query_index.rows), width=int(query_index.width),
                   system_name=getattr(query_index, "system_name", ""), source_path=str(query_index.path),
                   entries={q: [int(e.offset), int(e.length), bytes(e.seq_sha1).hex()]
                            for q, e in query_index.entries.items()},
                   sha256=AC.sha256_file(os.path.join(out, "queries.f16")))
        AC.write_json(enc, os.path.join(out, "query_encodings.json"))
        AC.write_json(dict(kw.get("query_sequences") or {}), os.path.join(out, "query_sequences.json"))
        if kw.get("stats") is None:
            kw["stats"] = {}
        opts = {}
        for kk, v in kw.items():
            if kk in ("stats", "query_sequences"):
                continue
            if kk == "output_needs" and not isinstance(v, (list, tuple)):
                import dataclasses
                v = dataclasses.asdict(v)
            opts[kk] = list(v) if isinstance(v, tuple) else v
        rec["stage2_capture_s"] = time.perf_counter() - t0
        t1 = time.perf_counter()
        res = orig_s2(query_index, prefilter_results, code_store, fetch, prepared, db_residues, pool, **kw)
        dt = time.perf_counter() - t1
        rec["stage2"] = dict(wall_s=dt, db_residues=int(db_residues), options=opts,
                             pool=dict(repr=repr(pool), workers=getattr(pool, "workers", None)),
                             code_store_bin=str(code_store.bin_path), queries_in=len(prefilter_results),
                             candidates_in=int(sum(len(v[0]) for v in prefilter_results.values())),
                             hits=int(sum(len(v) for v in res.values())), stats=kw["stats"],
                             system=dict(name=prepared.system.name, digest=prepared.system.digest()))
        # the hits at full precision (target index = the order of the database's code store ids is not needed: ids)
        tids = sorted({h.subject_id for hl in res.values() for h in hl})
        tpos = {t: i for i, t in enumerate(tids)}
        qids = list(prefilter_results)
        H = F.hits_arrays(res, qids, tpos)
        rec["hits_sha256"] = F.save_npz(os.path.join(out, "hits.npz"), qids=np.array(qids), tids=np.array(tids),
                                        **H)
        return res

    def write(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return orig_out(*args, **kwargs)
        finally:
            rec["phases"]["output_s"] = time.perf_counter() - t0

    EmbeddingComputer.in_input_order = in_input_order
    cli._filter_results_by_threshold = filter_by_threshold
    ca.align_candidates_esm_codes, output.write_aligned_results = stage2, write
    sys.argv = ["fm-search"] + fm_args
    rc = 0
    t_cli = time.perf_counter()
    try:
        cli.main()
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    finally:
        rec["cli_wall_s"] = time.perf_counter() - t_cli
        rec["rc"] = rc
        try:
            rec["provenance"] = AC.provenance(torch_info=True)
        except Exception as e:  # noqa: BLE001
            rec["provenance_error"] = repr(e)
            rec["provenance"] = AC.provenance(require_commit=None, torch_info=True)
        h = AC.write_json(rec, os.path.join(out, "run.json"))
        print(f"[a8_pipeline] rc={rc} cli_wall={rec['cli_wall_s']:.1f}s run.json sha256 {h}", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
