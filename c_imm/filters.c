/* Implementation of NLMS, single-mode KF, and IMM-KF. All linear-algebra
 * primitives go through linalg.h so the same source compiles against pure C
 * or OpenBLAS by relinking only.
 */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "linalg.h"
#include "filters.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* --- shared helper: shift tap-delay buffer one sample to the right --- */
static inline void shift_buf(double *buf, int L, double newest) {
    memmove(buf + 1, buf, (size_t)(L - 1) * sizeof(double));
    buf[0] = newest;
}

/* =====================================================================
 *  NLMS
 * ===================================================================== */
void nlms_init(nlms_t *f, int L, double mu) {
    f->L = L; f->mu = mu; f->eps = 1e-3;
    f->w   = calloc((size_t)L, sizeof(double));
    f->buf = calloc((size_t)L, sizeof(double));
}

double nlms_step(nlms_t *f, double xf, double d) {
    int L = f->L;
    shift_buf(f->buf, L, xf);
    double y = linalg_dot(L, f->w, f->buf);
    double e = d - y;
    double norm = linalg_dot(L, f->buf, f->buf) + f->eps;
    double step = f->mu * e / norm;
    linalg_axpy(L, step, f->buf, f->w);
    return e;
}

void nlms_free(nlms_t *f) {
    free(f->w); free(f->buf);
    f->w = f->buf = NULL;
}

/* =====================================================================
 *  Single-mode Kalman
 * ===================================================================== */
void kf_init(kf_t *f, int L, double sigma_q2, double sigma_r2) {
    f->L = L;
    f->sigma_q2 = sigma_q2;
    f->sigma_r2 = sigma_r2;
    f->S = sigma_r2;
    f->w      = calloc((size_t)L, sizeof(double));
    f->buf    = calloc((size_t)L, sizeof(double));
    f->P      = calloc((size_t)L * L, sizeof(double));
    f->P_pred = calloc((size_t)L * L, sizeof(double));
    f->Pxf    = calloc((size_t)L, sizeof(double));
    f->K      = calloc((size_t)L, sizeof(double));
    /* P0 = I */
    for (int i = 0; i < L; i++) f->P[(long)i * L + i] = 1.0;
}

double kf_step(kf_t *f, double xf, double d) {
    int L = f->L;
    shift_buf(f->buf, L, xf);

    /* P_pred = P + sigma_q2 * I */
    linalg_copy(L * L, f->P, f->P_pred);
    linalg_add_diag(f->P_pred, L, f->sigma_q2);

    /* Pxf = P_pred @ buf */
    linalg_matvec(f->P_pred, L, f->buf, f->Pxf);

    /* S  = buf . Pxf + sigma_r2 */
    double S = linalg_dot(L, f->buf, f->Pxf) + f->sigma_r2;

    /* K  = Pxf / S */
    linalg_copy(L, f->Pxf, f->K);
    linalg_scal(L, 1.0 / S, f->K);

    /* e  = d - w . buf */
    double e = d - linalg_dot(L, f->w, f->buf);

    /* w += K * e */
    linalg_axpy(L, e, f->K, f->w);

    /* P  = P_pred - outer(K, Pxf) */
    linalg_copy(L * L, f->P_pred, f->P);
    linalg_ger(f->P, L, -1.0, f->K, f->Pxf);

    f->S = S;
    return e;
}

void kf_free(kf_t *f) {
    free(f->w); free(f->buf); free(f->P);
    free(f->P_pred); free(f->Pxf); free(f->K);
    f->w = f->buf = f->P = f->P_pred = f->Pxf = f->K = NULL;
}

/* =====================================================================
 *  IMM-KF (M modes, vectorized stacked-state)
 *
 *  Mirrors src/imm.py step-by-step:
 *      1. Mixing (mu_cond, w_mix, P_mix)
 *      2. Per-mode KF update (P_pred, Pxf, S, K, innov, W, P)
 *      3. Mode posterior in log space, EWMA-smoothed
 *      4. Combination -> w, P_combined, e_combined, S
 * ===================================================================== */
void imm_init(imm_t *f, int L, int M,
              const double *Pi,
              const double *sigma_q2, const double *sigma_r2,
              int likelihood_window)
{
    f->L = L; f->M = M;
    f->alpha = 1.0 / (likelihood_window > 0 ? likelihood_window : 1);
    f->S = 0.0;

    f->Pi              = calloc((size_t)M * M, sizeof(double));
    f->sigma_q2        = calloc((size_t)M,     sizeof(double));
    f->sigma_r2        = calloc((size_t)M,     sizeof(double));
    f->W               = calloc((size_t)M * L, sizeof(double));
    f->P               = calloc((size_t)M * L * L, sizeof(double));
    f->mu              = calloc((size_t)M,     sizeof(double));
    f->log_lik_smooth  = calloc((size_t)M,     sizeof(double));
    f->w               = calloc((size_t)L,     sizeof(double));
    f->P_combined      = calloc((size_t)L * L, sizeof(double));
    f->buf             = calloc((size_t)L,     sizeof(double));
    f->cbar            = calloc((size_t)M,     sizeof(double));
    f->mu_cond         = calloc((size_t)M * M, sizeof(double));
    f->w_mix           = calloc((size_t)M * L, sizeof(double));
    f->P_mix           = calloc((size_t)M * L * L, sizeof(double));
    f->P_pred          = calloc((size_t)M * L * L, sizeof(double));
    f->Pxf             = calloc((size_t)M * L, sizeof(double));
    f->K               = calloc((size_t)M * L, sizeof(double));
    f->innov           = calloc((size_t)M,     sizeof(double));
    f->S_per           = calloc((size_t)M,     sizeof(double));
    f->diff_w          = calloc((size_t)L,     sizeof(double));

    memcpy(f->Pi,       Pi,       (size_t)M * M * sizeof(double));
    memcpy(f->sigma_q2, sigma_q2, (size_t)M * sizeof(double));
    memcpy(f->sigma_r2, sigma_r2, (size_t)M * sizeof(double));

    /* mu0 = 1/M each, P0 = I per mode, P_combined = I, S = mean(sigma_r2) */
    double inv_m = 1.0 / M;
    double s_mean = 0.0;
    for (int j = 0; j < M; j++) {
        f->mu[j] = inv_m;
        s_mean += sigma_r2[j];
        for (int i = 0; i < L; i++)
            f->P[(long)j * L * L + (long)i * L + i] = 1.0;
    }
    f->S = s_mean / M;
    for (int i = 0; i < L; i++) f->P_combined[(long)i * L + i] = 1.0;
}

double imm_step(imm_t *f, double xf, double d) {
    const int L = f->L, M = f->M;
    shift_buf(f->buf, L, xf);

    /* ----- 1. Mixing ----- */
    /* cbar[j] = sum_i Pi[i,j] * mu[i] */
    for (int j = 0; j < M; j++) {
        double s = 0.0;
        for (int i = 0; i < M; i++) s += f->Pi[(long)i * M + j] * f->mu[i];
        f->cbar[j] = s;
    }
    /* mu_cond[i,j] = Pi[i,j] * mu[i] / cbar[j] */
    for (int i = 0; i < M; i++) {
        double mi = f->mu[i];
        for (int j = 0; j < M; j++)
            f->mu_cond[(long)i * M + j] =
                f->Pi[(long)i * M + j] * mi / (f->cbar[j] + 1e-15);
    }
    /* w_mix[j,:] = sum_i mu_cond[i,j] * W[i,:] */
    memset(f->w_mix, 0, (size_t)M * L * sizeof(double));
    for (int j = 0; j < M; j++) {
        double *wm = f->w_mix + (long)j * L;
        for (int i = 0; i < M; i++) {
            double c = f->mu_cond[(long)i * M + j];
            const double *Wi = f->W + (long)i * L;
            linalg_axpy(L, c, Wi, wm);
        }
    }
    /* P_mix[j] = sum_i mu_cond[i,j] * (P[i] + outer(W[i]-w_mix[j])) */
    for (int j = 0; j < M; j++) {
        double *Pm = f->P_mix + (long)j * L * L;
        memset(Pm, 0, (size_t)L * L * sizeof(double));
        const double *wm = f->w_mix + (long)j * L;
        for (int i = 0; i < M; i++) {
            double c = f->mu_cond[(long)i * M + j];
            const double *Wi = f->W + (long)i * L;
            for (int k = 0; k < L; k++) f->diff_w[k] = Wi[k] - wm[k];
            /* Pm += c * P[i]  (loop fused) */
            const double *Pi_blk = f->P + (long)i * L * L;
            for (long k = 0; k < (long)L * L; k++) Pm[k] += c * Pi_blk[k];
            /* Pm += c * outer(diff, diff) */
            linalg_ger(Pm, L, c, f->diff_w, f->diff_w);
        }
    }

    /* ----- 2. Mode-conditioned KF update ----- */
    for (int j = 0; j < M; j++) {
        double *Pp = f->P_pred + (long)j * L * L;
        double *Pj = f->P      + (long)j * L * L;
        double *Pxf_j = f->Pxf + (long)j * L;
        double *K_j   = f->K   + (long)j * L;
        double *W_j   = f->W   + (long)j * L;
        const double *Pm = f->P_mix + (long)j * L * L;
        const double *wm = f->w_mix + (long)j * L;

        /* P_pred = P_mix + sigma_q2 * I */
        memcpy(Pp, Pm, (size_t)L * L * sizeof(double));
        linalg_add_diag(Pp, L, f->sigma_q2[j]);

        /* Pxf = P_pred @ buf */
        linalg_matvec(Pp, L, f->buf, Pxf_j);

        /* S = buf . Pxf + sigma_r2 */
        double S_j = linalg_dot(L, f->buf, Pxf_j) + f->sigma_r2[j];
        f->S_per[j] = S_j;

        /* K = Pxf / S */
        linalg_copy(L, Pxf_j, K_j);
        linalg_scal(L, 1.0 / S_j, K_j);

        /* innov = d - w_mix . buf */
        double innov_j = d - linalg_dot(L, wm, f->buf);
        f->innov[j] = innov_j;

        /* W[j] = w_mix + K * innov */
        linalg_copy(L, wm, W_j);
        linalg_axpy(L, innov_j, K_j, W_j);

        /* P[j] = P_pred - outer(K, Pxf) */
        linalg_copy(L * L, Pp, Pj);
        linalg_ger(Pj, L, -1.0, K_j, Pxf_j);
    }

    /* ----- 3. Mode probability update (smoothed log-likelihood) ----- */
    double ll_max = -1e300;
    for (int j = 0; j < M; j++) {
        double S_j = f->S_per[j];
        double iv = f->innov[j];
        double log_lik_j = -0.5 * (log(2.0 * M_PI * S_j) + iv * iv / S_j);
        f->log_lik_smooth[j] = (1.0 - f->alpha) * f->log_lik_smooth[j]
                             + f->alpha * log_lik_j;
        if (f->log_lik_smooth[j] > ll_max) ll_max = f->log_lik_smooth[j];
    }
    double post_sum = 0.0;
    for (int j = 0; j < M; j++) {
        double Lambda = exp(f->log_lik_smooth[j] - ll_max);
        f->mu[j] = Lambda * f->cbar[j];
        post_sum += f->mu[j];
    }
    double inv_psum = 1.0 / (post_sum + 1e-30);
    for (int j = 0; j < M; j++) f->mu[j] *= inv_psum;

    /* ----- 4. Combination ----- */
    /* w = sum_j mu[j] * W[j] */
    memset(f->w, 0, (size_t)L * sizeof(double));
    for (int j = 0; j < M; j++)
        linalg_axpy(L, f->mu[j], f->W + (long)j * L, f->w);

    /* P_combined = sum_j mu[j] * (P[j] + outer(W[j]-w)) */
    memset(f->P_combined, 0, (size_t)L * L * sizeof(double));
    for (int j = 0; j < M; j++) {
        double mj = f->mu[j];
        const double *W_j = f->W + (long)j * L;
        for (int k = 0; k < L; k++) f->diff_w[k] = W_j[k] - f->w[k];
        const double *Pj = f->P + (long)j * L * L;
        for (long k = 0; k < (long)L * L; k++) f->P_combined[k] += mj * Pj[k];
        linalg_ger(f->P_combined, L, mj, f->diff_w, f->diff_w);
    }

    /* e_combined = d - w . buf */
    double e_comb = d - linalg_dot(L, f->w, f->buf);

    /* S = sum_j mu[j] * (S_per[j] + (innov[j] - e_combined)^2) */
    double S_comb = 0.0;
    for (int j = 0; j < M; j++) {
        double de = f->innov[j] - e_comb;
        S_comb += f->mu[j] * (f->S_per[j] + de * de);
    }
    f->S = S_comb;
    return e_comb;
}

void imm_free(imm_t *f) {
    free(f->Pi); free(f->sigma_q2); free(f->sigma_r2);
    free(f->W); free(f->P); free(f->mu); free(f->log_lik_smooth);
    free(f->w); free(f->P_combined); free(f->buf);
    free(f->cbar); free(f->mu_cond);
    free(f->w_mix); free(f->P_mix); free(f->P_pred);
    free(f->Pxf); free(f->K);
    free(f->innov); free(f->S_per); free(f->diff_w);
    memset(f, 0, sizeof(*f));
}
