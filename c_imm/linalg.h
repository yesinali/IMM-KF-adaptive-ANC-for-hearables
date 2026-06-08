/* Minimal BLAS-style linear-algebra primitives used by the IMM/KF/NLMS
 * filters. Two backends provide the same prototypes:
 *   - linalg_pure.c : portable scalar C
 *   - linalg_blas.c : OpenBLAS (cblas_*)
 */
#ifndef LINALG_H
#define LINALG_H

/* y[i] = sum_j A[i,j] * x[j], A row-major, n x n */
void linalg_matvec(const double *A, int n, const double *x, double *y);

/* A[i,j] += alpha * x[i] * y[j]  (rank-1 update) */
void linalg_ger(double *A, int n, double alpha, const double *x, const double *y);

/* y[i] += alpha * x[i] */
void linalg_axpy(int n, double alpha, const double *x, double *y);

/* return sum_i x[i] * y[i] */
double linalg_dot(int n, const double *x, const double *y);

/* y[i] = x[i] */
void linalg_copy(int n, const double *x, double *y);

/* x[i] *= alpha */
void linalg_scal(int n, double alpha, double *x);

/* A[i,i] += d */
void linalg_add_diag(double *A, int n, double d);

/* y[i] += alpha * x[i] (vector form, length n) — same as axpy, kept for symmetry */

#endif /* LINALG_H */
