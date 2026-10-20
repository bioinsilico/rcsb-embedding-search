/* Self-hit study (v2probe.py): score-only Smith-Waterman on explicit float32 cell-score matrices.
 *
 * The recurrence of the research kernel's sw_score_f32 (embcomp/throughput/build/swkern.c), the DP of setting D:
 * 3-state Gotoh local alignment, cell score given (already z - offset), a gap of length L costs go + (L - 1) ge,
 * X <-> Y adjacency allowed, empty alignment 0. Not FoldMatch's kernel: v2probe.py checks it against FoldMatch's
 * S_f / S_r on the v1 cell scores before using it for the candidates.
 *
 *   cc -O3 -mcpu=native -shared -fPIC -o librawsw.dylib rawsw.c
 */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

static float sw_one(const float *S, int n, int m, int64_t ld, float go, float ge, float *H, float *X)
{
    const float NINF = -INFINITY;
    for (int j = 0; j <= m; j++) { H[j] = 0.0f; X[j] = NINF; }
    float best = 0.0f;
    for (int i = 1; i <= n; i++) {
        const float *s = S + (int64_t)(i - 1) * ld;
        float hdiag = 0.0f, vprev = 0.0f, y = NINF;
        for (int j = 1; j <= m; j++) {
            float hup = H[j];
            float xo = hup - go, xe = X[j] - ge;
            float x = xe > xo ? xe : xo;
            float mm = hdiag + s[j - 1];
            float v = x > mm ? x : mm;
            v = v <= 0.0f ? 0.0f : v;
            float yo = vprev - go, ye = y - ge;
            y = ye > yo ? ye : yo;
            float h = y > v ? y : v;
            hdiag = hup;
            H[j] = h; X[j] = x;
            vprev = v;
            if (h > best) best = h;
        }
    }
    return best;
}

/* K pairs side by side in one row-major [n, ld] matrix: pair k is columns col0[k] .. col0[k] + m[k] - 1. */
int sw_batch(const float *S, int n, int64_t ld, const int64_t *col0, const int32_t *m, int K, float go, float ge,
             float *out)
{
    int mmax = 0;
    for (int k = 0; k < K; k++) if (m[k] > mmax) mmax = m[k];
    float *H = (float *)malloc(sizeof(float) * (size_t)(mmax + 1));
    float *X = (float *)malloc(sizeof(float) * (size_t)(mmax + 1));
    if (!H || !X) { free(H); free(X); return -1; }
    for (int k = 0; k < K; k++) out[k] = sw_one(S + col0[k], n, m[k], ld, go, ge, H, X);
    free(H); free(X);
    return 0;
}
