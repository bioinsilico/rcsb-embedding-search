/*
 * census_swk.c -- P0.6 census driver over the UNMODIFIED research kernels (swkern.c #included verbatim, md5
 * 438d5a9811153f61e66cee845f1ed037, the evalue/score and throughput/build copy).
 *
 * census_multi_f32: one query x K candidate column blocks, ONE setting (offset, gap open, gap extend), both
 * directions.  For candidate k:
 *   research : rowcol_coef_f32 + ad_score on sim (S_f) and on simr (row-reversed copy; S_r) -- exactly the calls
 *              swz_ad_batch_f32 / swz_ad_multi_f32 make with tb = 0, so S_f / S_r are bit-identical to evalue/score.
 *   degenerate rows / columns: a row (column) whose population std is < 1e-6, detected from the coefficient
 *              a_i = 0.5 / (sd_i + 1e-8) >= 0.5 / (1e-6 + 1e-8) (b_j likewise); counted on the forward block.
 *   S7 variant (plan S7 'degenerate rule'): if any row / column is degenerate, its term contributes exactly 0
 *              (a_i = c_i = 0 for a row; b_j = 0, d_j = offset for a column) and the pair is re-scored; otherwise
 *              S7 == research.
 *   out  [K, 4] float32: S_f, S_r, S_f(S7), S_r(S7)
 *   deg  [K, 2] int32  : degenerate rows, degenerate columns (forward block)
 *   ws   floats >= 2 (n + mmax) + sw_ad_ws_floats(n, mmax)
 * Build (NERSC): gcc -O3 -march=znver3 -ffp-contract=off -shared -fPIC -o libcensus.so census_swk.c -lm
 */
#include "swkern.c"

#define A_DEG ((float)(0.5 / (1e-6 + 1e-8)))

static int s7_apply(int n, int m, float offset, float *ra, float *rc, float *cb, float *cd, int *ndr, int *ndc)
{
    int r = 0, c = 0;
    for (int i = 0; i < n; i++) if (ra[i] >= A_DEG) { ra[i] = 0.0f; rc[i] = 0.0f; r++; }
    for (int j = 0; j < m; j++) if (cb[j] >= A_DEG) { cb[j] = 0.0f; cd[j] = offset; c++; }
    if (ndr) *ndr = r;
    if (ndc) *ndc = c;
    return r + c;
}

void census_multi_f32(const float *sim, const float *simr, int n, int ld, const int64_t *col_off, int K,
                      float offset, float go, float ge, float *out, int32_t *deg, float *ws)
{
    int mmax = 0;
    for (int k = 0; k < K; k++) { int m = (int)(col_off[k + 1] - col_off[k]); if (m > mmax) mmax = m; }
    float *ra = ws, *rc = ws + n, *cb = ws + 2 * n, *cd = ws + 2 * n + mmax;
    float *dws = ws + 2 * (n + mmax);
    for (int k = 0; k < K; k++) {
        const int c0 = (int)col_off[k], m = (int)(col_off[k + 1] - col_off[k]);
        int bi, bj, ndr = 0, ndc = 0;
        /* forward */
        rowcol_coef_f32(sim + c0, n, m, ld, offset, ra, rc, cb, cd);
        float sf = ad_score(sim + c0, n, m, ld, ra, rc, cb, cd, go, ge, dws, &bi, &bj);
        float sf7 = sf;
        if (s7_apply(n, m, offset, ra, rc, cb, cd, &ndr, &ndc))
            sf7 = ad_score(sim + c0, n, m, ld, ra, rc, cb, cd, go, ge, dws, &bi, &bj);
        /* reversed query rows */
        rowcol_coef_f32(simr + c0, n, m, ld, offset, ra, rc, cb, cd);
        float sr = ad_score(simr + c0, n, m, ld, ra, rc, cb, cd, go, ge, dws, &bi, &bj);
        float sr7 = sr;
        if (s7_apply(n, m, offset, ra, rc, cb, cd, NULL, NULL))
            sr7 = ad_score(simr + c0, n, m, ld, ra, rc, cb, cd, go, ge, dws, &bi, &bj);
        out[4 * (size_t)k] = sf; out[4 * (size_t)k + 1] = sr; out[4 * (size_t)k + 2] = sf7; out[4 * (size_t)k + 3] = sr7;
        deg[2 * (size_t)k] = ndr; deg[2 * (size_t)k + 1] = ndc;
    }
}
