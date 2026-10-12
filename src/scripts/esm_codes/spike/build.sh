#!/bin/bash
# P0.4 spike: build every fmsw_spike.c variant for this host plus the research kernels, into OUT_DIR.
#   build.sh OUT_DIR RESEARCH_SRC_DIR
#     RESEARCH_SRC_DIR must hold the research swkern.c (md5 438d5a9811153f61e66cee845f1ed037) and swmulti.c
#     (md5 b12fbc2c03638659440b12282a4d8670); they are compiled unmodified with the research flags.
#   Env: GCC (default gcc, gcc-14 on macOS), CLANG (default clang, cc on macOS), TILES=1 adds similarity-tile variants
#        (primary compiler only).
# Variants (name:flags, commas = spaces):
#   x86_64  sse2 (baseline, xor similarity), avx2 (-mavx2, xor), avx2fma (-mavx2 -mfma, fma similarity),
#           avx2hyb (-mavx2 -mfma -DSIM_HYBRID, 3x3 tile: 2 fma + 1 xor-add column vectors), generic
#   arm64   neon (fma-by-lane similarity), neonxor (-DSIM_FORCE_XOR), generic
# Output: OUT_DIR/libfmsw_<compiler>_<variant>[_rR_cC].<ext>, OUT_DIR/research/libswkern.<ext>, libswmulti.<ext>,
#         OUT_DIR/build_manifest.txt (compiler versions, flags, md5 of every source).
set -euo pipefail
OUT=${1:?OUT_DIR}; RSRC=${2:?RESEARCH_SRC_DIR}
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=$HERE/fmsw_spike.c
mkdir -p "$OUT/research"
OS=$(uname -s); ARCH=$(uname -m)
BASE="-O3 -std=c11 -fPIC -shared -ffp-contract=off -fno-fast-math -fvisibility=hidden"
GCC_EXTRA=""
if [[ $OS == Darwin ]]; then
  EXT=dylib; GCC=${GCC:-gcc-14}; CLANG=${CLANG:-cc}
  GCC_EXTRA="-isysroot $(xcrun --show-sdk-path)"          # homebrew gcc-14 needs the SDK sysroot on macOS 26
  VARIANTS="neon: neonxor:-DSIM_FORCE_XOR generic:-DFMSW_GENERIC"
  RFLAGS="-O3 -mcpu=apple-m1 -ffp-contract=off -shared -fPIC"; RCC=cc; PRIMARY=clang
  TILESET="neon:4:4 neon:8:2 neon:6:3 neon:4:3 neon:3:4 neon:8:3"
else
  EXT=so; GCC=${GCC:-gcc}; CLANG=${CLANG:-clang}; PRIMARY=gcc
  if [[ $ARCH == x86_64 ]]; then
    VARIANTS="sse2: avx2:-mavx2 avx2fma:-mavx2,-mfma avx2hyb:-mavx2,-mfma,-DSIM_HYBRID,-DSIM_RB=3,-DSIM_CB=3 generic:-DFMSW_GENERIC"
    RFLAGS="-O3 -march=znver3 -ffp-contract=off -shared -fPIC"; RCC=gcc
    TILESET="avx2fma:6:2 avx2fma:4:3 avx2fma:3:4 avx2fma:8:1 avx2:6:2 avx2:4:3 sse2:4:2 sse2:4:3"
  else
    VARIANTS="neon: neonxor:-DSIM_FORCE_XOR generic:-DFMSW_GENERIC"
    RFLAGS="-O3 -mcpu=native -ffp-contract=off -shared -fPIC"; RCC=gcc
    TILESET="neon:4:4 neon:8:2 neon:6:3"
  fi
fi
flags_of() { local v=$1; for X in $VARIANTS; do [[ ${X%%:*} == "$v" ]] && { echo "${X#*:}" | tr ',' ' '; return; }; done; }
md5of() { (md5sum "$1" 2>/dev/null || md5 -r "$1") | cut -d' ' -f1; }
M=$OUT/build_manifest.txt
{
  echo "host $(hostname) $OS $ARCH $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "md5 fmsw_spike.c $(md5of "$SRC")"
  echo "md5 research swkern.c $(md5of "$RSRC/swkern.c")"
  echo "md5 research swmulti.c $(md5of "$RSRC/swmulti.c")"
  echo "base flags: $BASE"
} > "$M"
build() {   # cc name variant extra_flags suffix
  local CC=$1 NAME=$2 V=$3 FL=$4 SUF=$5 XF=""
  [[ $NAME == gcc ]] && XF=$GCC_EXTRA
  local O="$OUT/libfmsw_${NAME}_${V}${SUF}.$EXT"
  $CC $BASE $XF $FL -o "$O" "$SRC" -lm 2>&1 | grep -v "overriding-deployment" || true
  [[ -f "$O" ]] || { echo "build failed: $O"; exit 1; }
  echo "built $(basename "$O") flags: $BASE $XF $FL" >> "$M"
}
for CCPAIR in "gcc=$GCC" "clang=$CLANG"; do
  NAME=${CCPAIR%%=*}; CC=${CCPAIR#*=}
  if ! command -v "$CC" >/dev/null 2>&1; then echo "skip $NAME ($CC not found)" | tee -a "$M"; continue; fi
  echo "$NAME: $CC = $($CC --version 2>&1 | head -1)" >> "$M"
  for X in $VARIANTS; do
    V=${X%%:*}
    build "$CC" "$NAME" "$V" "$(flags_of "$V")" ""
  done
  if [[ ${TILES:-0} == 1 && $NAME == "$PRIMARY" ]]; then
    for T in $TILESET; do
      IFS=: read -r V RB CB <<< "$T"
      build "$CC" "$NAME" "$V" "$(flags_of "$V") -DSIM_RB=$RB -DSIM_CB=$CB" "_r${RB}_c${CB}"
    done
  fi
done
$RCC $RFLAGS -o "$OUT/research/libswkern.$EXT" "$RSRC/swkern.c" -lm
$RCC $RFLAGS -I"$RSRC" -o "$OUT/research/libswmulti.$EXT" "$RSRC/swmulti.c" -lm
echo "research: $RCC $RFLAGS ($($RCC --version 2>&1 | head -1))" >> "$M"
cat "$M"
