/* C port benchmark driver.
 *
 * Reads a scenario dumped by `scripts/c_port_dump.py`, runs NLMS / KF / IMM
 * through the entire sample stream, times each filter, and writes the
 * residuals out so a sister Python script can compare numerics against the
 * reference NumPy implementation.
 *
 * Binary input format (little-endian):
 *   int32   N           number of samples
 *   int32   L           FIR length
 *   int32   M           number of IMM modes (== 4)
 *   int32   FS          sample rate
 *   int32   W_lik       likelihood window (samples)
 *   float64 mu_nlms     NLMS step size
 *   float64 Q_kf, R_kf  single-mode KF (sigma_q2, sigma_r2)
 *   float64 Pi[M*M]     transition matrix
 *   float64 Q_imm[M]    per-mode sigma_q2
 *   float64 R_imm[M]    per-mode sigma_r2
 *   float64 xf[N]       filtered reference
 *   float64 d[N]        primary
 *
 * Binary output format:
 *   int32   N
 *   int32   M
 *   float64 e_nlms[N], e_kf[N], e_imm[N]
 *   float64 mu_hist[N * M]    IMM mode posterior, row-major (k stride M)
 *
 * Timing JSON is printed to stdout.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include "filters.h"

#ifndef BACKEND_LABEL
#define BACKEND_LABEL "unknown"
#endif

static double now_sec(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static void *xmalloc(size_t bytes) {
    void *p = malloc(bytes);
    if (!p) { fprintf(stderr, "out of memory (%zu bytes)\n", bytes); exit(2); }
    return p;
}

static void read_exact(void *buf, size_t bytes, FILE *fp, const char *what) {
    if (fread(buf, 1, bytes, fp) != bytes) {
        fprintf(stderr, "short read on %s\n", what);
        exit(2);
    }
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s input.bin output.bin\n", argv[0]);
        return 1;
    }
    FILE *fi = fopen(argv[1], "rb");
    if (!fi) { perror(argv[1]); return 1; }

    int32_t N, L, M, FS, W_lik;
    read_exact(&N,  4, fi, "N");
    read_exact(&L,  4, fi, "L");
    read_exact(&M,  4, fi, "M");
    read_exact(&FS, 4, fi, "FS");
    read_exact(&W_lik, 4, fi, "W_lik");

    double mu_nlms, Q_kf, R_kf;
    read_exact(&mu_nlms, 8, fi, "mu_nlms");
    read_exact(&Q_kf,    8, fi, "Q_kf");
    read_exact(&R_kf,    8, fi, "R_kf");

    double *Pi    = xmalloc((size_t)M * M * sizeof(double));
    double *Q_imm = xmalloc((size_t)M * sizeof(double));
    double *R_imm = xmalloc((size_t)M * sizeof(double));
    read_exact(Pi,    (size_t)M * M * 8, fi, "Pi");
    read_exact(Q_imm, (size_t)M * 8,     fi, "Q_imm");
    read_exact(R_imm, (size_t)M * 8,     fi, "R_imm");

    double *xf = xmalloc((size_t)N * sizeof(double));
    double *d  = xmalloc((size_t)N * sizeof(double));
    read_exact(xf, (size_t)N * 8, fi, "xf");
    read_exact(d,  (size_t)N * 8, fi, "d");
    fclose(fi);

    double *e_nlms   = xmalloc((size_t)N * sizeof(double));
    double *e_kf     = xmalloc((size_t)N * sizeof(double));
    double *e_imm    = xmalloc((size_t)N * sizeof(double));
    double *mu_hist  = xmalloc((size_t)N * (size_t)M * sizeof(double));

    double audio_sec = (double)N / (double)FS;

    /* ---- NLMS ---- */
    nlms_t nlms; nlms_init(&nlms, L, mu_nlms);
    double t0 = now_sec();
    for (int k = 0; k < N; k++) e_nlms[k] = nlms_step(&nlms, xf[k], d[k]);
    double t_nlms = now_sec() - t0;
    nlms_free(&nlms);

    /* ---- KF ---- */
    kf_t kf; kf_init(&kf, L, Q_kf, R_kf);
    t0 = now_sec();
    for (int k = 0; k < N; k++) e_kf[k] = kf_step(&kf, xf[k], d[k]);
    double t_kf = now_sec() - t0;
    kf_free(&kf);

    /* ---- IMM (with mode-posterior history dump) ---- */
    imm_t imm;
    imm_init(&imm, L, M, Pi, Q_imm, R_imm, W_lik);
    t0 = now_sec();
    for (int k = 0; k < N; k++) {
        e_imm[k] = imm_step(&imm, xf[k], d[k]);
        memcpy(&mu_hist[(size_t)k * M], imm.mu, (size_t)M * sizeof(double));
    }
    double t_imm = now_sec() - t0;
    imm_free(&imm);

    /* ---- Write residuals + IMM mu history ---- */
    FILE *fo = fopen(argv[2], "wb");
    if (!fo) { perror(argv[2]); return 1; }
    fwrite(&N, 4, 1, fo);
    fwrite(&M, 4, 1, fo);
    fwrite(e_nlms, sizeof(double), (size_t)N, fo);
    fwrite(e_kf,   sizeof(double), (size_t)N, fo);
    fwrite(e_imm,  sizeof(double), (size_t)N, fo);
    fwrite(mu_hist, sizeof(double), (size_t)N * (size_t)M, fo);
    fclose(fo);

    /* ---- Timing JSON ---- */
    printf("{\n");
    printf("  \"backend\": \"%s\",\n", BACKEND_LABEL);
    printf("  \"N\": %d,\n", N);
    printf("  \"L\": %d,\n", L);
    printf("  \"M\": %d,\n", M);
    printf("  \"FS\": %d,\n", FS);
    printf("  \"audio_sec\": %.6f,\n", audio_sec);
    printf("  \"t_nlms_sec\": %.6e,\n", t_nlms);
    printf("  \"t_kf_sec\":   %.6e,\n", t_kf);
    printf("  \"t_imm_sec\":  %.6e,\n", t_imm);
    printf("  \"rtf_nlms\":   %.6f,\n", t_nlms / audio_sec);
    printf("  \"rtf_kf\":     %.6f,\n", t_kf   / audio_sec);
    printf("  \"rtf_imm\":    %.6f,\n", t_imm  / audio_sec);
    printf("  \"per_sample_us_nlms\": %.4f,\n", 1e6 * t_nlms / N);
    printf("  \"per_sample_us_kf\":   %.4f,\n", 1e6 * t_kf   / N);
    printf("  \"per_sample_us_imm\":  %.4f\n",  1e6 * t_imm  / N);
    printf("}\n");

    free(Pi); free(Q_imm); free(R_imm);
    free(xf); free(d);
    free(e_nlms); free(e_kf); free(e_imm); free(mu_hist);
    return 0;
}
