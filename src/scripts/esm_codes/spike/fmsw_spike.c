/*
 * fmsw_spike.c -- P0.4 kernel spike for FoldMatch esm-codes-v1 (research prototype, NOT the P3 kernel).
 *
 * Derived from the research kernel embcomp/throughput/build/swkern.c (md5 438d5a9811153f61e66cee845f1ed037):
 * traceback(), interleave_strip(), ad_core() and the rowcol_coef two-pass structure are copied with these changes
 *   1. row strides are int64 (ptrdiff_t) so a candidate block can be read through a ROW-REVERSED VIEW
 *      (base = last row, stride = -ld); nothing else in the DP changes;
 *   2. rowcol reductions always use 4-lane vectors (the SSE / NEON order of the research kernel), on every ISA;
 *   3. S7 degenerate rule: a row (column) with population std < 1e-6 gets coefficients 0 (column: d_j = offset);
 *   4. a generic path (-DFMSW_GENERIC) with GCC/Clang vector extensions only; vmax(a, b) = a > b ? a : b per lane
 *      (the operand order of _mm_max_ps);
 *   5. new fixed-order query_prep (S5) and similarity (S6) in C, so the whole chain q fp16 -> S_f, S_r, path is
 *      bit-identical on every ISA / compiler (no BLAS).
 *
 *   fmsw_query_prep   qp_i = fp32(fp64(q_i) / (8 sqrt(sum_k fp64(q_ik)^2))), k ascending in float64; 0 if |q_i| < 1e-12
 *   fmsw_sim_codes    s_ij = +0 + sum_{k=0..63, ascending} (bit_jk ? qp_ik : -qp_ik), float32 accumulator per (i, j);
 *                     vector lanes run over j only (the sign flip is an XOR of the sign bit, which is exactly -x)
 *   fmsw_score_fwd_rev  per candidate: rowcol coef + score-only strip DP on the block (S_f) and on the row-reversed
 *                     view of the same block (S_r); no simr buffer
 *   fmsw_align        per candidate: forward DP with traceback (research swz_ad_batch_f32 tb = 1)
 *   fmsw_block_fwd_rev  sim_codes into a caller buffer + score_fwd_rev (one ctypes call per block)
 *   fmsw_list_fwd_rev   the same for a whole candidate list, blocks of <= block_floats similarity floats (all in C)
 *
 * Build (see build.sh): cc -O3 -std=c11 -fPIC -shared -ffp-contract=off -fno-fast-math -fvisibility=hidden
 *   [-mavx2 | -DFMSW_GENERIC] [-DSIM_RB=r -DSIM_CB=c] -o libfmsw_<isa>.so fmsw_spike.c -lm
 */
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define API __attribute__((visibility("default")))

#define F_HY 1   /* H[i,j] = Y[i,j]            */
#define F_VX 2   /* V[i,j] = X[i,j]            */
#define F_XE 4   /* X[i,j] extends X[i-1,j]    */
#define F_YE 8   /* Y[i,j] extends Y[i,j-1]    */
#define F_ST 16  /* V[i,j] <= 0: start here    */
#define WS 16    /* rows per anti-diagonal strip */

/* ------------------------------------------------------------------------------------------------ */
/* ISA selection                                                                                     */
/* ------------------------------------------------------------------------------------------------ */
#if defined(FMSW_GENERIC)
#define VL 4
#define ISA_NAME "generic"
typedef float vf __attribute__((vector_size(16)));
typedef int32_t vi __attribute__((vector_size(16)));
typedef uint8_t vb __attribute__((vector_size(4)));
static inline vf vmax(vf a, vf b) { vi m = (vi)(a > b); return (vf)(((vi)a & m) | ((vi)b & ~m)); }
#define SHIFTIN(p, c) __builtin_shufflevector((p), (c), 3, 4, 5, 6)
#define LANES {0, 1, 2, 3}
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#define VL 4
#define ISA_NAME "neon"
typedef float vf __attribute__((vector_size(16)));
typedef int32_t vi __attribute__((vector_size(16)));
typedef uint8_t vb __attribute__((vector_size(4)));
static inline vf vmax(vf a, vf b) { return (vf)vmaxq_f32((float32x4_t)a, (float32x4_t)b); }
#define SHIFTIN(p, c) __builtin_shufflevector((p), (c), 3, 4, 5, 6)
#define LANES {0, 1, 2, 3}
#elif defined(__AVX2__)
#include <immintrin.h>
#define VL 8
#define ISA_NAME "avx2"
typedef float vf __attribute__((vector_size(32)));
typedef int32_t vi __attribute__((vector_size(32)));
typedef uint8_t vb __attribute__((vector_size(8)));
static inline vf vmax(vf a, vf b) { return (vf)_mm256_max_ps((__m256)a, (__m256)b); }
#define SHIFTIN(p, c) __builtin_shufflevector((p), (c), 7, 8, 9, 10, 11, 12, 13, 14)
#define LANES {0, 1, 2, 3, 4, 5, 6, 7}
#elif defined(__SSE2__)
#include <immintrin.h>
#define VL 4
#define ISA_NAME "sse2"
typedef float vf __attribute__((vector_size(16)));
typedef int32_t vi __attribute__((vector_size(16)));
typedef uint8_t vb __attribute__((vector_size(4)));
static inline vf vmax(vf a, vf b) { return (vf)_mm_max_ps((__m128)a, (__m128)b); }
#define SHIFTIN(p, c) __builtin_shufflevector((p), (c), 3, 4, 5, 6)
#define LANES {0, 1, 2, 3}
#else
#error "no vector ISA: build with -DFMSW_GENERIC"
#endif
#define NVEC (WS / VL)

/* similarity tile: SIM_RB query rows x SIM_CB vectors of VL target columns */
#ifndef SIM_RB
#if defined(__ARM_NEON) && !defined(FMSW_GENERIC)
#define SIM_RB 4
#elif VL == 8
#define SIM_RB 4
#else
#define SIM_RB 6
#endif
#endif
#ifndef SIM_CB
#if defined(__ARM_NEON) && !defined(FMSW_GENERIC)
#define SIM_CB 4
#else
#define SIM_CB 2
#endif
#endif
#define SIM_TW (SIM_CB * VL)

typedef float f4 __attribute__((vector_size(16)));
static inline vf vload(const float *p) { vf v; memcpy(&v, p, sizeof(vf)); return v; }
static inline void vstore(float *p, vf v) { memcpy(p, &v, sizeof(vf)); }
static inline vf vbc(float x) { vf z = {0}; return z + x; }
static inline vi vsel_i(vi mask, vi a, vi b) { return (a & mask) | (b & ~mask); }
static inline f4 ld4(const float *p) { f4 v; memcpy(&v, p, 16); return v; }
static inline void st4(float *p, f4 v) { memcpy(p, &v, 16); }
static inline f4 bc4(float x) { f4 z = {0, 0, 0, 0}; return z + x; }
static inline float hsum4(f4 v) { float s = 0.0f; s += v[0]; s += v[1]; s += v[2]; s += v[3]; return s; }

API int fmsw_abi_version(void) { return 0; }                 /* spike: 0 (P3 kernel will be 1) */
API const char *fmsw_isa(void) { return ISA_NAME; }
API int fmsw_vector_lanes(void) { return VL; }
API int fmsw_sim_tile(void) { return SIM_RB * 1000 + SIM_TW; }
API const char *fmsw_compiler(void)
{
#if defined(__clang__)
    return "clang " __clang_version__;
#elif defined(__GNUC__)
    return "gcc " __VERSION__;
#else
    return "unknown";
#endif
}

/* ------------------------------------------------------------------------------------------------ */
/* S5 query_prep                                                                                     */
/* ------------------------------------------------------------------------------------------------ */
API int fmsw_query_prep(const float *q, int n, float *qp)
{
    if (n < 0 || (n > 0 && (!q || !qp))) return 2;
    for (int i = 0; i < n; i++) {
        const float *r = q + (size_t)i * 64;
        float *o = qp + (size_t)i * 64;
        double ss = 0.0;
        for (int k = 0; k < 64; k++) {
            const double x = (double)r[k];
            if (!isfinite(x)) return 2;
            ss += x * x;                        /* exact product (24 x 24 bits), one rounding per add */
        }
        const double nrm = sqrt(ss);
        if (nrm < 1e-12) { memset(o, 0, 64 * sizeof(float)); continue; }
        const double den = 8.0 * nrm;
        for (int k = 0; k < 64; k++) o[k] = (float)((double)r[k] / den);
    }
    return 0;
}

/* ------------------------------------------------------------------------------------------------ */
/* S6 similarity                                                                                     */
/*   codes: uint8 [*, 8] packbits MSB-first (bit k of residue j = codes[j][k >> 3] >> (7 - (k & 7)) & 1);  */
/*   sim[i * ld + (j - j_begin)] for j in [col_off[0], col_off[K])                                     */
/*   Per column tile a table mt[64][SIM_TW] is built once from the codes and reused for every query row: */
/*     xor mode:  mt = 0 (bit 1) / 0x80000000 (bit 0);  acc_ij += qp_ik XOR mt_jk  (== bit ? qp : -qp)    */
/*     fma modes: mt = +1.0f (bit 1) / -1.0f (bit 0);   acc_ij = fma(qp_ik, mt_jk, acc_ij)               */
/*   Both are the same IEEE operation: the product qp * (+-1) is exact, so the fused multiply-add rounds  */
/*   exactly once, like the plain add of +-qp. Per (i, j) the terms are added for k = 0..63 in order.     */
/*   fma-neon (every aarch64 build) uses FMLA by element (4 k per loaded query vector); fma-avx2 is used   */
/*   when the build has __FMA__ (-mavx2 -mfma); SSE2 and the generic build use xor.                      */
/* ------------------------------------------------------------------------------------------------ */
#if !defined(SIM_FORCE_XOR) && !defined(FMSW_GENERIC) && defined(__ARM_NEON) && defined(__aarch64__)
#define SIM_MODE 1
#define SIM_MODE_NAME "fma-neon-lane"
#elif !defined(SIM_FORCE_XOR) && !defined(FMSW_GENERIC) && defined(__AVX2__) && defined(__FMA__) && defined(SIM_HYBRID)
#define SIM_MODE 3
#define SIM_MODE_NAME "hybrid-avx2"
#elif !defined(SIM_FORCE_XOR) && !defined(FMSW_GENERIC) && defined(__AVX2__) && defined(__FMA__)
#define SIM_MODE 2
#define SIM_MODE_NAME "fma-avx2"
#else
#define SIM_MODE 0
#define SIM_MODE_NAME "xor-add"
#endif
API const char *fmsw_sim_mode(void) { return SIM_MODE_NAME; }

static const float ZROW[64] = {0};

/* hybrid-avx2 (SIM_MODE 3): the first SIM_CB - 1 column vectors use FMA (FP0/FP1 pipes on Zen 3), the last one
   uses xor + add (FADD pipes), so both pipe pairs work; same per-cell operation order */
static inline void sim_tile_masks(const uint8_t *codes, int64_t j0, int jw, uint32_t *mt)
{
    for (int c = 0; c < SIM_TW; c++) {
        const uint32_t base = (SIM_MODE == 0 || (SIM_MODE == 3 && c >= (SIM_CB - 1) * VL)) ? 0u : 0x3f800000u;
        uint64_t w = ~(uint64_t)0;                                  /* padding lanes: never stored */
        if (c < jw) {
            const uint8_t *b = codes + (size_t)(j0 + c) * 8;
            w = ((uint64_t)b[0] << 56) | ((uint64_t)b[1] << 48) | ((uint64_t)b[2] << 40) | ((uint64_t)b[3] << 32)
              | ((uint64_t)b[4] << 24) | ((uint64_t)b[5] << 16) | ((uint64_t)b[6] << 8) | (uint64_t)b[7];
        }
        const uint64_t nw = ~w;                                      /* bit k of nw (from the top) = negate */
        for (int k = 0; k < 64; k++) mt[k * SIM_TW + c] = base | ((uint32_t)((nw >> (63 - k)) & 1u) << 31);
    }
}

/* the accumulators live in a local array (address never escapes, so they stay in registers) and are copied
   to the caller's array once at the end: writing them through the out pointer inside the k loop made GCC store
   every accumulator on every k step (vf may alias the float query rows) */
static inline void sim_tile_rows(const float *const *qr, const uint32_t *restrict mt, vf out[SIM_RB][SIM_CB])
{
    vf acc[SIM_RB][SIM_CB];
    for (int r = 0; r < SIM_RB; r++)
        for (int c = 0; c < SIM_CB; c++) acc[r][c] = (vf){0};
#if SIM_MODE == 1
    for (int kq = 0; kq < 16; kq++) {
        float32x4_t qv[SIM_RB];
        for (int r = 0; r < SIM_RB; r++) qv[r] = vld1q_f32(qr[r] + 4 * kq);
#define FMSW_LANE(L)                                                                                       \
        {                                                                                                  \
            const int k = 4 * kq + (L);                                                                    \
            float32x4_t d[SIM_CB];                                                                         \
            for (int c = 0; c < SIM_CB; c++) d[c] = vld1q_f32((const float *)(mt + k * SIM_TW + c * VL));  \
            for (int r = 0; r < SIM_RB; r++)                                                               \
                for (int c = 0; c < SIM_CB; c++)                                                           \
                    acc[r][c] = (vf)vfmaq_laneq_f32((float32x4_t)acc[r][c], d[c], qv[r], (L));             \
        }
        FMSW_LANE(0) FMSW_LANE(1) FMSW_LANE(2) FMSW_LANE(3)
#undef FMSW_LANE
    }
#elif SIM_MODE == 2
    for (int k = 0; k < 64; k++) {
        __m256 d[SIM_CB];
        for (int c = 0; c < SIM_CB; c++) d[c] = _mm256_loadu_ps((const float *)(mt + k * SIM_TW + c * VL));
        for (int r = 0; r < SIM_RB; r++) {
            const __m256 qb = _mm256_broadcast_ss(qr[r] + k);
            for (int c = 0; c < SIM_CB; c++) acc[r][c] = (vf)_mm256_fmadd_ps(qb, d[c], (__m256)acc[r][c]);
        }
    }
#elif SIM_MODE == 3
    for (int k = 0; k < 64; k++) {
        __m256 d[SIM_CB - 1];
        for (int c = 0; c < SIM_CB - 1; c++) d[c] = _mm256_loadu_ps((const float *)(mt + k * SIM_TW + c * VL));
        const __m256 mx = _mm256_loadu_ps((const float *)(mt + k * SIM_TW + (SIM_CB - 1) * VL));
        for (int r = 0; r < SIM_RB; r++) {
            const __m256 qb = _mm256_broadcast_ss(qr[r] + k);
            for (int c = 0; c < SIM_CB - 1; c++) acc[r][c] = (vf)_mm256_fmadd_ps(qb, d[c], (__m256)acc[r][c]);
            acc[r][SIM_CB - 1] = (vf)_mm256_add_ps((__m256)acc[r][SIM_CB - 1], _mm256_xor_ps(qb, mx));
        }
    }
#else
    for (int k = 0; k < 64; k++) {
        vi m[SIM_CB];
        for (int c = 0; c < SIM_CB; c++) memcpy(&m[c], mt + k * SIM_TW + c * VL, sizeof(vi));
        for (int r = 0; r < SIM_RB; r++) {
            const vi qb = (vi)vbc(qr[r][k]);
            for (int c = 0; c < SIM_CB; c++) acc[r][c] += (vf)(qb ^ m[c]);
        }
    }
#endif
    memcpy(out, acc, sizeof(acc));
}

API int fmsw_sim_codes(const float *qp, int n, const uint8_t *codes, const int64_t *col_off, int K,
                       float *sim, int64_t ld)
{
    if (n <= 0 || K <= 0) return (n < 0 || K < 0) ? 2 : 0;
    const int64_t jb = col_off[0], je = col_off[K];
    if (je < jb || ld < je - jb) return 2;
    uint32_t mt[64 * SIM_TW] __attribute__((aligned(64)));
    for (int64_t j0 = jb; j0 < je; j0 += SIM_TW) {
        const int jw = (int)((je - j0) < SIM_TW ? (je - j0) : SIM_TW);
        sim_tile_masks(codes, j0, jw, mt);
        float *out = sim + (j0 - jb);
        for (int i0 = 0; i0 < n; i0 += SIM_RB) {
            const float *qr[SIM_RB];
            for (int r = 0; r < SIM_RB; r++) qr[r] = (i0 + r < n) ? qp + (size_t)(i0 + r) * 64 : ZROW;
            vf acc[SIM_RB][SIM_CB];
            sim_tile_rows(qr, mt, acc);
            const int rows = (n - i0) < SIM_RB ? (n - i0) : SIM_RB;
            for (int r = 0; r < rows; r++) {
                float *o = out + (ptrdiff_t)(i0 + r) * ld;
                if (jw == SIM_TW) {
                    for (int c = 0; c < SIM_CB; c++) vstore(o + c * VL, acc[r][c]);
                } else {
                    for (int j = 0; j < jw; j++) o[j] = acc[r][j / VL][j % VL];
                }
            }
        }
    }
    return 0;
}

/* ------------------------------------------------------------------------------------------------ */
/* rowcol z coefficients: research rowcol_coef_f32 with the 4-lane (SSE / NEON) order on every ISA,  */
/* int64 strides (negative = reversed view) and the S7 degenerate rule                              */
/* ------------------------------------------------------------------------------------------------ */
#define DEGEN_SD 1e-6
static void rowcol_coef4(const float *sim, int n, int m, int64_t ld, float offset,
                         float *ra, float *rc, float *cb, float *cd)
{
    for (int j = 0; j < m; j++) cb[j] = 0.0f;
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        const float *s0 = sim + (ptrdiff_t)i * ld, *s1 = s0 + ld, *s2 = s1 + ld, *s3 = s2 + ld;
        f4 a0 = {0}, a1 = {0}, a2 = {0}, a3 = {0};
        int j = 0;
        for (; j + 4 <= m; j += 4) {
            f4 v0 = ld4(s0 + j), v1 = ld4(s1 + j), v2 = ld4(s2 + j), v3 = ld4(s3 + j);
            a0 += v0; a1 += v1; a2 += v2; a3 += v3;
            st4(cb + j, ld4(cb + j) + ((v0 + v1) + (v2 + v3)));
        }
        float r0 = hsum4(a0), r1 = hsum4(a1), r2 = hsum4(a2), r3 = hsum4(a3);
        for (; j < m; j++) { r0 += s0[j]; r1 += s1[j]; r2 += s2[j]; r3 += s3[j]; cb[j] += (s0[j] + s1[j]) + (s2[j] + s3[j]); }
        ra[i] = (float)((double)r0 / m); ra[i + 1] = (float)((double)r1 / m);
        ra[i + 2] = (float)((double)r2 / m); ra[i + 3] = (float)((double)r3 / m);
    }
    for (; i < n; i++) {
        const float *s = sim + (ptrdiff_t)i * ld;
        f4 acc = {0};
        int j = 0;
        for (; j + 4 <= m; j += 4) { f4 v = ld4(s + j); acc += v; st4(cb + j, ld4(cb + j) + v); }
        float r = hsum4(acc);
        for (; j < m; j++) { r += s[j]; cb[j] += s[j]; }
        ra[i] = (float)((double)r / m);
    }
    const float inv_n = (float)(1.0 / n);
    for (int j = 0; j < m; j++) cb[j] *= inv_n;
    for (int j = 0; j < m; j++) cd[j] = 0.0f;
    i = 0;
    for (; i + 4 <= n; i += 4) {
        const float *s0 = sim + (ptrdiff_t)i * ld, *s1 = s0 + ld, *s2 = s1 + ld, *s3 = s2 + ld;
        const f4 m0 = bc4(ra[i]), m1 = bc4(ra[i + 1]), m2 = bc4(ra[i + 2]), m3 = bc4(ra[i + 3]);
        f4 a0 = {0}, a1 = {0}, a2 = {0}, a3 = {0};
        int j = 0;
        for (; j + 4 <= m; j += 4) {
            f4 v0 = ld4(s0 + j), v1 = ld4(s1 + j), v2 = ld4(s2 + j), v3 = ld4(s3 + j);
            f4 d0 = v0 - m0, d1 = v1 - m1, d2 = v2 - m2, d3 = v3 - m3;
            a0 += d0 * d0; a1 += d1 * d1; a2 += d2 * d2; a3 += d3 * d3;
            f4 mc = ld4(cb + j);
            f4 e0 = v0 - mc, e1 = v1 - mc, e2 = v2 - mc, e3 = v3 - mc;
            st4(cd + j, ld4(cd + j) + ((e0 * e0 + e1 * e1) + (e2 * e2 + e3 * e3)));
        }
        float r0 = hsum4(a0), r1 = hsum4(a1), r2 = hsum4(a2), r3 = hsum4(a3);
        for (; j < m; j++) {
            float d0 = s0[j] - ra[i], d1 = s1[j] - ra[i + 1], d2 = s2[j] - ra[i + 2], d3 = s3[j] - ra[i + 3];
            r0 += d0 * d0; r1 += d1 * d1; r2 += d2 * d2; r3 += d3 * d3;
            float e0 = s0[j] - cb[j], e1 = s1[j] - cb[j], e2 = s2[j] - cb[j], e3 = s3[j] - cb[j];
            cd[j] += (e0 * e0 + e1 * e1) + (e2 * e2 + e3 * e3);
        }
        rc[i] = r0; rc[i + 1] = r1; rc[i + 2] = r2; rc[i + 3] = r3;
    }
    for (; i < n; i++) {
        const float *s = sim + (ptrdiff_t)i * ld;
        const float mr = ra[i];
        const f4 vmr = bc4(mr);
        f4 acc = {0};
        int j = 0;
        for (; j + 4 <= m; j += 4) {
            f4 v = ld4(s + j);
            f4 dr = v - vmr, dc = v - ld4(cb + j);
            acc += dr * dr;
            st4(cd + j, ld4(cd + j) + dc * dc);
        }
        float r = hsum4(acc);
        for (; j < m; j++) { float dr = s[j] - mr, dc = s[j] - cb[j]; r += dr * dr; cd[j] += dc * dc; }
        rc[i] = r;
    }
    for (int k = 0; k < n; k++) {
        const double sd = sqrt((double)rc[k] / m);
        if (sd < DEGEN_SD) { ra[k] = 0.0f; rc[k] = 0.0f; continue; }
        double a = 0.5 / (sd + 1e-8);
        rc[k] = (float)(a * ra[k]);
        ra[k] = (float)a;
    }
    for (int j = 0; j < m; j++) {
        const double sd = sqrt((double)cd[j] / n);
        if (sd < DEGEN_SD) { cb[j] = 0.0f; cd[j] = offset; continue; }
        double b = 0.5 / (sd + 1e-8);
        cd[j] = (float)(b * cb[j] + offset);
        cb[j] = (float)b;
    }
}

/* ------------------------------------------------------------------------------------------------ */
/* traceback (research, unchanged)                                                                    */
/* ------------------------------------------------------------------------------------------------ */
static inline uint8_t flag_strip(const uint8_t *fl, int m, int i, int j)
{
    if (i <= 0 || j <= 0) return F_ST;
    const int T = m + WS;
    const int s = (i - 1) / WS, r = (i - 1) % WS;
    return fl[((size_t)s * T + (size_t)(j + r)) * WS + r];
}

static int traceback(const uint8_t *fl, int m, int bi, int bj, int32_t *path)
{
    int i = bi, j = bj, k = 0;
    int state = 0; /* 0 H, 1 V, 2 M, 3 X, 4 Y */
    for (;;) {
        uint8_t f = flag_strip(fl, m, i, j);
        if (state == 0) {
            state = (f & F_HY) ? 4 : 1;
        } else if (state == 1) {
            if (f & F_ST) break;
            state = (f & F_VX) ? 3 : 2;
        } else if (state == 2) {
            path[2 * k] = i - 1; path[2 * k + 1] = j - 1; k++;
            i--; j--; state = 0;
        } else if (state == 3) {
            path[2 * k] = i - 1; path[2 * k + 1] = -1; k++;
            state = (f & F_XE) ? 3 : 0;
            i--;
        } else {
            path[2 * k] = -1; path[2 * k + 1] = j - 1; k++;
            state = (f & F_YE) ? 4 : 1;
            j--;
        }
    }
    for (int a = 0, b = k - 1; a < b; a++, b--) {
        int32_t t0 = path[2 * a], t1 = path[2 * a + 1];
        path[2 * a] = path[2 * b]; path[2 * a + 1] = path[2 * b + 1];
        path[2 * b] = t0; path[2 * b + 1] = t1;
    }
    return k;
}

/* ------------------------------------------------------------------------------------------------ */
/* anti-diagonal strip DP (research, row stride int64)                                               */
/* ------------------------------------------------------------------------------------------------ */
#define BIG_NEG (-1e30f)
#define PADW(m) ((m) + 48)
static size_t ad_ws_floats(int m) { return (size_t)2 * (m + WS + 1) + (size_t)(m + WS + 4) * WS + (size_t)WS * PADW(m) + 64; }

API size_t fmsw_ws_floats(int n, int64_t mmax) { return (size_t)2 * ((size_t)n + (size_t)mmax) + ad_ws_floats((int)mmax); }
API size_t fmsw_flag_bytes(int n, int64_t mmax) { return (size_t)((n + WS - 1) / WS) * (size_t)(mmax + WS) * WS; }

static void interleave_strip(const float *S, int n, int m, int64_t ld, int i0,
                             const float *ra, const float *rc, const float *cb, const float *cd,
                             float *P, float *Sv)
{
    const int PW = PADW(m);
    const int T = m + WS;
    const f4 z4 = {0, 0, 0, 0};
    for (int r = 0; r < WS; r++) {
        float *pr = P + (size_t)r * PW;
        const int i = i0 + r;
        st4(pr, z4); st4(pr + 4, z4); st4(pr + 8, z4); st4(pr + 12, z4);
        float *d = pr + 16;
        int c = 0;
        if (i <= n) {
            const float *s = S + (ptrdiff_t)(i - 1) * ld;
            const float a = ra[i - 1], cc = rc[i - 1];
            const f4 a4 = {a, a, a, a}, c4 = {cc, cc, cc, cc};
            for (; c + 4 <= m; c += 4) st4(d + c, ld4(s + c) * (a4 + ld4(cb + c)) - (c4 + ld4(cd + c)));
            for (; c < m; c++) d[c] = s[c] * (a + cb[c]) - (cc + cd[c]);
        }
        for (; c < PW - 16; c++) d[c] = BIG_NEG;
    }
    for (int g = 0; g < WS / 4; g++) {
        const float *p0 = P + (size_t)(4 * g + 0) * PW + 15 - (4 * g + 0);
        const float *p1 = P + (size_t)(4 * g + 1) * PW + 15 - (4 * g + 1);
        const float *p2 = P + (size_t)(4 * g + 2) * PW + 15 - (4 * g + 2);
        const float *p3 = P + (size_t)(4 * g + 3) * PW + 15 - (4 * g + 3);
        float *out = Sv + 4 * g;
        for (int t0 = 0; t0 < T; t0 += 4) {
            const f4 R0 = ld4(p0 + t0), R1 = ld4(p1 + t0), R2 = ld4(p2 + t0), R3 = ld4(p3 + t0);
            const f4 a0 = __builtin_shufflevector(R0, R1, 0, 4, 1, 5), a1 = __builtin_shufflevector(R0, R1, 2, 6, 3, 7);
            const f4 a2 = __builtin_shufflevector(R2, R3, 0, 4, 1, 5), a3 = __builtin_shufflevector(R2, R3, 2, 6, 3, 7);
            st4(out + (size_t)(t0 + 0) * WS, __builtin_shufflevector(a0, a2, 0, 1, 4, 5));
            st4(out + (size_t)(t0 + 1) * WS, __builtin_shufflevector(a0, a2, 2, 3, 6, 7));
            st4(out + (size_t)(t0 + 2) * WS, __builtin_shufflevector(a1, a3, 0, 1, 4, 5));
            st4(out + (size_t)(t0 + 3) * WS, __builtin_shufflevector(a1, a3, 2, 3, 6, 7));
        }
    }
}

static inline __attribute__((always_inline))
float ad_core(const float *S, int n, int m, int64_t ld,
              const float *ra, const float *rc, const float *cb, const float *cd,
              float go, float ge, float *ws, uint8_t *fl, const int tb, int *bi_out, int *bj_out)
{
    const float NINF = -INFINITY;
    const int T = m + WS;
    float *Hrow = ws, *Xrow = ws + (m + WS + 1);
    float *Sv = ws + 2 * (m + WS + 1);
    float *P = Sv + (size_t)(m + WS + 4) * WS;
    for (int j = 0; j <= m + WS; j++) { Hrow[j] = 0.0f; Xrow[j] = NINF; }
    float best = 0.0f;
    int bi = 0, bj = 0;
    const vf vgo = vbc(go), vge = vbc(ge), zero = {0}, ninf = vbc(NINF);
    const vi lanes = LANES;
    const int nstrip = (n + WS - 1) / WS;
    for (int st = 0; st < nstrip; st++) {
        const int i0 = 1 + st * WS;
        interleave_strip(S, n, m, ld, i0, ra, rc, cb, cd, P, Sv);
        vf H1[NVEC], X1[NVEC], V1[NVEC], Y1[NVEC], HU[NVEC], lbest[NVEC];
        vi lbj[NVEC], tj[NVEC];
        for (int k = 0; k < NVEC; k++) {
            H1[k] = zero; X1[k] = ninf; V1[k] = zero; Y1[k] = ninf; HU[k] = zero;
            lbest[k] = vbc(best);
            lbj[k] = (vi){0} - 1;
            tj[k] = ((vi){0} + (1 - VL * k)) - lanes;
        }
        uint8_t *fs = tb ? fl + (size_t)st * T * WS : NULL;
        for (int t = 1; t < T; t++) {
            const float *sv = Sv + (size_t)t * WS;
            const vf hu0 = vbc(Hrow[t]), xu0 = vbc(Xrow[t]);
            for (int k = NVEC - 1; k >= 0; k--) {
                const vf hup = SHIFTIN(k ? H1[k - 1] : hu0, H1[k]);
                const vf xup = SHIFTIN(k ? X1[k - 1] : xu0, X1[k]);
                const vf hdg = HU[k];
                HU[k] = hup;
                const vf xo = hup - vgo, xe = xup - vge;
                const vf x = vmax(xo, xe);
                const vf mm = hdg + vload(sv + VL * k);
                const vf v0 = vmax(x, mm);
                const vf v = vmax(v0, zero);
                const vf yo = V1[k] - vgo, ye = Y1[k] - vge;
                const vf y = vmax(yo, ye);
                const vf h = vmax(v, y);
                if (tb) {
                    vi f = ((xe > xo) & F_XE) | ((x > mm) & F_VX) | ((v0 <= zero) & F_ST) | ((ye > yo) & F_YE)
                           | ((y > v) & F_HY);
                    vb fb = __builtin_convertvector(f, vb);
                    memcpy(fs + (size_t)t * WS + VL * k, &fb, VL);
                }
                const vi gt = h > lbest[k];
                lbest[k] = vmax(lbest[k], h);
                lbj[k] = vsel_i(gt, tj[k], lbj[k]);
                tj[k] += 1;
                H1[k] = h; X1[k] = x; V1[k] = v; Y1[k] = y;
            }
            int jl = t - (WS - 1);
            jl = jl > 0 ? jl : 0;
            Hrow[jl] = H1[NVEC - 1][VL - 1];
            Xrow[jl] = X1[NVEC - 1][VL - 1];
        }
        for (int r = 0; r < WS && i0 + r <= n; r++) {
            const int k = r / VL, l = r % VL;
            if (lbj[k][l] >= 1 && lbest[k][l] > best) { best = lbest[k][l]; bi = i0 + r; bj = lbj[k][l]; }
        }
    }
    *bi_out = bi; *bj_out = bj;
    return best + 0.0f;                         /* -0 -> +0 (S8) */
}

static float ad_score(const float *S, int n, int m, int64_t ld, const float *ra, const float *rc, const float *cb,
                      const float *cd, float go, float ge, float *ws, int *bi, int *bj)
{
    return ad_core(S, n, m, ld, ra, rc, cb, cd, go, ge, ws, NULL, 0, bi, bj);
}

static float ad_tb(const float *S, int n, int m, int64_t ld, const float *ra, const float *rc, const float *cb,
                   const float *cd, float go, float ge, float *ws, uint8_t *fl, int *bi, int *bj)
{
    return ad_core(S, n, m, ld, ra, rc, cb, cd, go, ge, ws, fl, 1, bi, bj);
}

static int64_t block_mmax(const int64_t *col_off, int K)
{
    int64_t mmax = 0;
    for (int k = 0; k < K; k++) { int64_t m = col_off[k + 1] - col_off[k]; if (m > mmax) mmax = m; }
    return mmax;
}

/* ------------------------------------------------------------------------------------------------ */
/* S9 score-only S_f and S_r from ONE similarity block; S_r reads the row-reversed view              */
/*   candidate k = columns col_off[k] .. col_off[k+1] of sim [n, ld] (col_off relative to sim)        */
/*   ends_f / ends_r (optional): 1-based best cell, (0, 0) = empty                                     */
/* ------------------------------------------------------------------------------------------------ */
API int fmsw_score_fwd_rev(const float *sim, int n, int64_t ld, const int64_t *col_off, int K, float offset,
                           float go, float ge, float *s_fwd, float *s_rev, int32_t *ends_f, int32_t *ends_r,
                           float *ws, size_t ws_len)
{
    if (n <= 0 || K < 0) return 2;
    const int64_t mmax = block_mmax(col_off, K);
    if (ws_len < fmsw_ws_floats(n, mmax)) return 1;
    float *ra = ws, *rc = ws + n, *cb = ws + 2 * n, *cd = ws + 2 * n + mmax;
    float *dws = ws + 2 * ((size_t)n + (size_t)mmax);
    const float *last = sim + (ptrdiff_t)(n - 1) * ld;
    for (int k = 0; k < K; k++) {
        const int64_t c0 = col_off[k];
        const int m = (int)(col_off[k + 1] - c0);
        int bi, bj;
        if (m <= 0) { s_fwd[k] = s_rev[k] = 0.0f; continue; }
        rowcol_coef4(sim + c0, n, m, ld, offset, ra, rc, cb, cd);
        s_fwd[k] = ad_score(sim + c0, n, m, ld, ra, rc, cb, cd, go, ge, dws, &bi, &bj);
        if (ends_f) { ends_f[2 * k] = bi; ends_f[2 * k + 1] = bj; }
        rowcol_coef4(last + c0, n, m, -ld, offset, ra, rc, cb, cd);
        s_rev[k] = ad_score(last + c0, n, m, -ld, ra, rc, cb, cd, go, ge, dws, &bi, &bj);
        if (ends_r) { ends_r[2 * k] = bi; ends_r[2 * k + 1] = bj; }
    }
    return 0;
}

/* forward run with traceback (research swz_ad_batch_f32, tb = 1); rev = 1 runs it on the reversed view instead.
   path_buf int32 [2 * path_cap]; path_off int64 [K + 1]                                                        */
API int fmsw_align(const float *sim, int n, int64_t ld, const int64_t *col_off, int K, float offset, float go,
                   float ge, int rev, float *scores, int32_t *ends, int32_t *path_buf, size_t path_cap,
                   int64_t *path_off, float *ws, size_t ws_len, uint8_t *flags, size_t flags_len)
{
    if (n <= 0 || K < 0) return 2;
    const int64_t mmax = block_mmax(col_off, K);
    if (ws_len < fmsw_ws_floats(n, mmax) || flags_len < fmsw_flag_bytes(n, mmax)) return 1;
    float *ra = ws, *rc = ws + n, *cb = ws + 2 * n, *cd = ws + 2 * n + mmax;
    float *dws = ws + 2 * ((size_t)n + (size_t)mmax);
    const float *base = rev ? sim + (ptrdiff_t)(n - 1) * ld : sim;
    const int64_t lds = rev ? -ld : ld;
    int64_t po = 0;
    path_off[0] = 0;
    for (int k = 0; k < K; k++) {
        const int64_t c0 = col_off[k];
        const int m = (int)(col_off[k + 1] - c0);
        int bi = 0, bj = 0, plen = 0;
        float sc = 0.0f;
        if (m > 0) {
            if ((size_t)(po + n + m) > path_cap) return 1;
            rowcol_coef4(base + c0, n, m, lds, offset, ra, rc, cb, cd);
            sc = ad_tb(base + c0, n, m, lds, ra, rc, cb, cd, go, ge, dws, flags, &bi, &bj);
            plen = traceback(flags, m, bi, bj, path_buf + 2 * po);
        }
        scores[k] = sc;
        ends[2 * k] = bi; ends[2 * k + 1] = bj;
        po += plen;
        path_off[k + 1] = po;
    }
    return 0;
}

/* one block: sim (caller buffer, ld = col_off[K] - col_off[0]) + fwd/rev scores; col_off absolute into codes */
API int fmsw_block_fwd_rev(const float *qp, int n, const uint8_t *codes, const int64_t *col_off, int K,
                           float offset, float go, float ge, float *s_fwd, float *s_rev,
                           float *simbuf, size_t sim_len, float *ws, size_t ws_len, int64_t *co_rel)
{
    const int64_t w = col_off[K] - col_off[0];
    if ((size_t)n * (size_t)w > sim_len) return 1;
    int st = fmsw_sim_codes(qp, n, codes, col_off, K, simbuf, w);
    if (st) return st;
    for (int k = 0; k <= K; k++) co_rel[k] = col_off[k] - col_off[0];
    return fmsw_score_fwd_rev(simbuf, n, w, co_rel, K, offset, go, ge, s_fwd, s_rev, NULL, NULL, ws, ws_len);
}

/* whole candidate list (col_off absolute into codes, K candidates), greedy blocks of <= block_floats sim floats
   (the research _blocks rule: a candidate starts a new block if the current one is non-empty and would overflow). */
API int fmsw_list_fwd_rev(const float *qp, int n, const uint8_t *codes, const int64_t *col_off, int K,
                          float offset, float go, float ge, int64_t block_floats, float *s_fwd, float *s_rev,
                          float *simbuf, size_t sim_len, float *ws, size_t ws_len, int64_t *co_tmp)
{
    int k0 = 0;
    while (k0 < K) {
        int k1 = k0 + 1;
        int64_t acc = (int64_t)n * (col_off[k1] - col_off[k0]);
        while (k1 < K && acc + (int64_t)n * (col_off[k1 + 1] - col_off[k1]) <= block_floats) {
            acc += (int64_t)n * (col_off[k1 + 1] - col_off[k1]);
            k1++;
        }
        int st = fmsw_block_fwd_rev(qp, n, codes, col_off + k0, k1 - k0, offset, go, ge, s_fwd + k0, s_rev + k0,
                                    simbuf, sim_len, ws, ws_len, co_tmp);
        if (st) return st;
        k0 = k1;
    }
    return 0;
}
