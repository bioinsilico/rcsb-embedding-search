# blas_div: BLAS-divergence study harness (FoldMatch esm-codes-v1, decision D2)

Question: if stage 2 computes the similarity `sim = query_prep(q) . decode(codes)^T` with a BLAS SGEMM (D2 option b,
the research path) instead of the fixed-order C loop (option a), how far can `S_f`, `S_r`, `S`, `Q90`, `S'`, `lnE`
and the reported alignment diverge between builds (BLAS library and version, core type, threads, device)?

Only the similarity changes between backends. As a variant, the fp32 numpy query norm can change too. Rowcol z, the
setting-D Smith-Waterman (forward with traceback, reverse on the row-reversed view), Q90 and E always run in the same
fixed-order spike kernel (`../spike/fmsw_spike.c` via `../spike/spike_lib.py`) and in float64 numpy.

- **Reference:** `--backend fixed_c --qprep c`. This is the spike `fmsw_query_prep` + `fmsw_sim_codes`, which gives
  identical bits on every build; `verify golden` reproduces the P0.4 golden sha256 `5204a97a...`.
- **Oracle:** float64 sim, z and DP. `compare` uses it for the margins of differing paths and for the accuracy of the
  similarity.

## Files

- `blas_div.py`: the whole harness. Run `python blas_div.py -h`; the module docstring holds the details.
- `ROOT` is `/pscratch/sd/j/jsegura/embcomp/blasdiv` on NERSC and `<scratchpad>/blasdiv` on the Mac. Set `BLASDIV_ROOT`
  to override it.
  - `ROOT/data/`: inputs, built once on NERSC by `prep` and rsynced to the Mac. `MANIFEST.json` lists the md5s.
    - research copies: queries, targets, panel codes, `ship_final.json`, labels, research Q90;
    - `q150.json`: the shared 150-query subset (25 per M3 length bin, seed 20261012);
    - `a1.npz`: 12,228 CATH domain pairs and 732 chain pairs, with q fp16, codes, US-align refs and the stored
      setting-D alignments;
    - `research_SD_q150.npz`: research `S_D_fwd`/`S_D_rev` rows;
    - `spike_inputs/`: the golden-hash inputs.
  - `ROOT/spike/build/`: spike libraries built by `spike/build.sh`. The default is `libfmsw_gcc_avx2hyb.so` on x86_64 and
    `libfmsw_clang_neon.dylib` on arm64. Override with `--lib` or `BLASDIV_LIB_DIR`.
  - `ROOT/runs/<host-tag>/<work>/<label>/`: `meta.json` (host, CPU, env incl. `OPENBLAS_CORETYPE`/`MKL_*`, numpy BLAS
    via `np.show_config` and threadpoolctl, torch and device info, spike lib md5, timings), `search.npz` (`Sf`, `Sr`
    float32 [nq, NT], panel `Pf`, `Pr` [nq, 1000]) or `a1.npz` (`Sf`, `Sr`, traceback score, end cell, forward path),
    `simblocks.npz` (full similarity of sampled blocks) and `repeat.json`.
  - `ROOT/compare/*.json|npz`: `compare` outputs.

## Subcommands

| Command | What it does |
|---|---|
| `info [--torch]` | Prints a JSON report: host, BLAS, torch, env, spike library and backend availability. |
| `prep` | Builds `ROOT/data` from the research files. Run it once, on NERSC. |
| `sim --backend B [--qprep c\|numpy] --work search\|a1 ...` | Computes the similarity and scores it straight away. |
| `compare RUN_A RUN_B` | Measures how far B diverges from A (A = reference). |
| `verify golden [--backend B]` | Checks the golden hash. With another backend, it reports per-pair differences against `fixed_c`. |
| `verify run RUN` | For `fixed_c` runs, checks against spike's own C drivers. For any backend, checks against the research stored scores and stored setting-D alignments. |

### What `compare` reports

- per-pair `S_f`, `S_r`, `S`, `S'`, `lnE`: fraction bit-equal, abs / rel / ulp distributions;
- `Q90` and the panel;
- E-threshold flips (also split by TP / real-null label), `E <= 1e-3` and `E <= 1` hit lists, top-10 and top-100 order;
- for A1: path identity, differing columns, F1 against US-align and the stored alignments, and the float64-oracle
  best-cell and traceback margins of every differing path;
- for sampled blocks: sim cells bit-equal, max abs, ulp, and the fraction correctly rounded against float64.

## Backends

- `fixed_c`
- `numpy`: `qp @ D.T`, exactly as research `score.py` did.
- `numpy_rowwise`: one GEMM per target.
- `torch_cpu`
- `torch_cuda`: TF32 off, `highest` precision.
- `torch_cuda_tf32`: a hazard demonstration.
- `torch_mps`

An unavailable backend makes the command exit with status 2 and the message `BACKEND UNAVAILABLE`.

`--repeat all` recomputes every block with variants `same`, `off4`, `off8`, `off16`, `ld` and `dT` and compares the
bits. These are the misaligned buffer, padded leading-dimension and materialised-transpose layouts.

## Work sets

- **`search`:** queries × 50,000 benchmark targets, plus the 1,000 panel members. Choose the queries with
  `--queries q150` (the default), `all`, `q150:K`, `first:N` or `rows:...`, and limit the targets with `--tmax N`.
  - Each query's targets go into blocks of `--block-floats` (2^17) similarity floats, using the production rule.
  - `--block-rule research --block-floats 1048576` uses research `score.py`'s chunking instead.
- **`a1`:** all A1 pairs, one GEMM per pair. `--a1-limit N` gives a stratified slice.
- **Parallelism:** `--workers W` spawned processes, each with `--threads T` BLAS or torch threads (default 1). The caller
  sets `OPENBLAS_CORETYPE`, `MKL_CBWR`, `MKL_DEBUG_CPU_TYPE` and the like, and the run records them.

## E-values

`compare` uses the `ship_final.json` table `D_fmr/pnl_uniref50/M3t999` (`all`):

- `Q90 = np.quantile(f64 P_f - f64 P_r, 0.9)`
- `S' = S - Q90`
- `lnE = lnK(m) + ln m + ln N - lambda(m) S'`, with `np.interp` in ln m (clamped)
- `N = 8,557,236` annotated benchmark residues. This is ship.py's `N_ANN` over all 50,000 columns, which equals
  `ship_final` meta `N_ann_fit_db`.
