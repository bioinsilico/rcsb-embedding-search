#!/bin/bash
# Self-hit study of esm-codes-v1 (SELFHIT_REPORT.md): the commands, in order, as run on 2026-10-19 (M1 Max).
# FoldMatch is used read-only from the P6 sandbox tree; nothing in FoldMatch or the esm-codes-v1 registry changes.
# ESM3 runs (step 2) go through heavy.sh: one model-loading process at a time, >= 20 GB free.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SB=/Users/joan/devel/foldmatch-p2p3-sandbox
DATA=$SB/work/selfhit_data                 # inputs, encodings, scores (not in the repo)
source $SB/work/p6fin/env.sh               # PYTHONPATH=wt-P6/src (tree a9584eb), HF offline cache, PY, HEAVY
export TMPDIR=$DATA/tmp
mkdir -p $DATA/{inputs,enc,score,probe,tmp}

# 0. data (copied from NERSC, read-only): db_u10k/ (P6 check database: P.rescodes.*, P.sequences, P.metadata),
#    u10k/uniref50_10k.fasta, scop/queries.L2m_B8_s0.npz (calibration benchmark queries), census/items.tsv
#    (P0.6 census items), p6/queries200.fasta; the P6 outputs are in $SB/p6_nersc/.

# 1. inputs
$PY -I $HERE/make_inputs.py $DATA
$PY -I $HERE/make_inputs.py $DATA --pdb

# 2. ESM3 + esm-codes-v1 head (FoldMatch path), query encodings AND target codes, MPS
for s in u50_q200 u50_db800 scop_q pdb_r600; do
  $HEAVY $PY $HERE/encode_fm.py --fasta $DATA/inputs/$s.fasta --out $DATA/enc/$s.npz --device mps > $DATA/enc/$s.log 2>&1
done

# 3. self pair + panel with FoldMatch's kernel, float64 self-matrix diagnostics
$PY $HERE/selfscore.py $DATA/enc/u50_q200.npz $DATA/score/u50_q200.npz --workers 4 --db $DATA/db_u10k
$PY $HERE/selfscore.py $DATA/enc/u50_db800.npz $DATA/score/u50_db800.npz --workers 6 --db $DATA/db_u10k
$PY $HERE/selfscore.py $DATA/enc/scop_q.npz $DATA/score/scop_q.npz --workers 6
$PY $HERE/selfscore.py $DATA/enc/pdb_r600.npz $DATA/score/pdb_r600.npz --workers 8

# 4. code homogeneity and the panel background (closed form)
$PY $HERE/bgprobe.py $DATA $DATA/probe/bg.npz

# 5. tables and figures
$PY $HERE/analyze.py $DATA $SB/p6_nersc/results/p6-chk-q1_60008416/runs $HERE/results

# 6. FUTURE WORK probe: candidate cell scores with a mini-recalibration on random UniRef50 nulls
$PY $HERE/v2probe.py $DATA $DATA/probe/v2probe.npz --workers 8 > $DATA/probe/v2probe.log 2>&1
$PY $HERE/v2probe.py --analyze $DATA/probe/v2probe.npz $DATA > $DATA/probe/v2probe_analyze.log
cp $DATA/probe/v2probe_summary.json $HERE/results/
