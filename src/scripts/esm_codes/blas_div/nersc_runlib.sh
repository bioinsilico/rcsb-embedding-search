# nersc_runlib.sh -- sourced by /global/homes/j/jsegura/jobs/blasdiv-*.sbatch (NERSC CPU runs of the BLAS-divergence
# study; EPYC 7763 nodes).  Copy lives at /pscratch/sd/j/jsegura/embcomp/blasdiv/nersc_runlib.sh.
#
#   brun PY BDIR WORK QUERIES LABEL "ENV=.. ENV=.." [blas_div.py sim args ...]
#     -> ROOT/runs/epyc/BDIR/<search-QUERIES-T50000 | a1-all>/LABEL   (skipped when its search.npz / a1.npz exists;
#        an interrupted run resumes from its parts/).  After the run: blas_probe.json (same python, same env, same
#        --threads; OPENBLAS_VERBOSE=2 core line in blas_probe.stderr) and verify.json (background; call `wait`).
#   bcmp REF TEST [compare args ...]   -> ROOT/compare/<auto name>.json (skipped when the json exists)
ROOT=/pscratch/sd/j/jsegura/embcomp/blasdiv
HT=epyc7763
FM_PY=/global/homes/j/jsegura/.conda/envs/foldmatch/bin/python
RUNS=$ROOT/runs/epyc
export PYTHONDONTWRITEBYTECODE=1
cd $ROOT || exit 1

rdir() {  # rdir BDIR WORK QUERIES LABEL
  if [ "$2" = a1 ]; then echo "$RUNS/$1/a1-all/$4"; else echo "$RUNS/$1/search-$3-T50000/$4"; fi
}

brun() {
  local py=$1 bdir=$2 work=$3 q=$4 label=$5 envs=$6
  shift 6
  local out fin thr=1 tflag="" a
  out=$(rdir "$bdir" "$work" "$q" "$label")
  if [ "$work" = a1 ]; then fin=a1.npz; else fin=search.npz; fi
  if [ -f "$out/$fin" ]; then echo "SKIP (done) $out"; return 0; fi
  local prev=""
  for a in "$@"; do
    [ "$prev" = --threads ] && thr=$a
    case "$a" in torch_*) tflag="--torch" ;; esac
    prev=$a
  done
  echo "=== $(date '+%F %T') RUN $out | env: [$envs] | py: $py | args: $*"
  local t0=$SECONDS
  env $envs "$py" blas_div.py sim --work "$work" --queries "$q" --host-tag $HT --label "$label" --out "$out" "$@"
  local rc=$?
  echo "=== $(date '+%F %T') rc=$rc $((SECONDS - t0))s $out"
  [ $rc -ne 0 ] && return $rc
  env $envs OPENBLAS_VERBOSE=2 "$py" blas_probe.py --threads "$thr" $tflag --out "$out/blas_probe.json" \
    > /dev/null 2> "$out/blas_probe.stderr"
  "$FM_PY" blas_div.py verify run "$out" > "$out/verify.log" 2>&1 &
  return 0
}

bcmp() {  # bcmp REF_RUN_DIR TEST_RUN_DIR [args]; name = harness default
  local A=$1 B=$2
  shift 2
  "$FM_PY" blas_div.py compare "$A" "$B" "$@" > "$ROOT/logs/cmp.$(basename "$(dirname "$A")").$(basename "$A").VS.$(basename "$B").$$.$RANDOM.log" 2>&1
}
