/* NLMS, single-mode KF, and IMM-KF filter implementations.
 *
 * API mirrors src/filters.py and src/imm.py. Each filter maintains its own
 * internal tap-delay buffer, so the caller only feeds one (xf, d) sample
 * per step and receives the residual e back.
 *
 * All filters expose .S (innovation variance, last step) for consistency
 * checks; KF additionally exposes .P, IMM exposes .P_combined.
 */
#ifndef FILTERS_H
#define FILTERS_H

typedef struct {
    int L;
    double mu, eps;
    double *w;     /* (L,)  filter weights */
    double *buf;   /* (L,)  tap-delay buffer */
} nlms_t;

typedef struct {
    int L;
    double sigma_q2, sigma_r2;
    double S;       /* last innovation variance */
    double *w;      /* (L,) state mean */
    double *P;      /* (L,L) posterior covariance, row-major */
    double *buf;    /* (L,) tap buffer */
    /* workspaces */
    double *P_pred; /* (L,L) */
    double *Pxf;    /* (L,) */
    double *K;      /* (L,) */
} kf_t;

typedef struct {
    int L, M;
    double alpha;            /* smoothing rate = 1/W */
    double S;                /* combined innovation variance, last step */
    /* per-mode */
    double *Pi;              /* (M,M) row-major transition matrix */
    double *sigma_q2;        /* (M,) */
    double *sigma_r2;        /* (M,) */
    double *W;               /* (M,L) per-mode mean */
    double *P;               /* (M,L,L) per-mode covariance */
    double *mu;              /* (M,) mode posterior */
    double *log_lik_smooth;  /* (M,) EWMA of per-mode log-likelihood */
    /* combined */
    double *w;               /* (L,) */
    double *P_combined;      /* (L,L) */
    double *buf;             /* (L,) tap buffer */
    /* workspaces */
    double *cbar;     /* (M,) */
    double *mu_cond;  /* (M,M) */
    double *w_mix;    /* (M,L) */
    double *P_mix;    /* (M,L,L) */
    double *P_pred;   /* (M,L,L) */
    double *Pxf;      /* (M,L) */
    double *K;        /* (M,L) */
    double *innov;    /* (M,) */
    double *S_per;    /* (M,) */
    double *diff_w;   /* (L,) */
} imm_t;

/* NLMS */
void nlms_init(nlms_t *f, int L, double mu);
double nlms_step(nlms_t *f, double xf, double d);
void nlms_free(nlms_t *f);

/* Single-mode KF */
void kf_init(kf_t *f, int L, double sigma_q2, double sigma_r2);
double kf_step(kf_t *f, double xf, double d);
void kf_free(kf_t *f);

/* IMM-KF (M modes) */
void imm_init(imm_t *f, int L, int M,
              const double *Pi,
              const double *sigma_q2, const double *sigma_r2,
              int likelihood_window);
double imm_step(imm_t *f, double xf, double d);
void imm_free(imm_t *f);

#endif /* FILTERS_H */
