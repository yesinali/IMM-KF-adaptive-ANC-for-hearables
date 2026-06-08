/* OpenBLAS backend: thin wrappers over cblas. Same prototypes as
 * linalg_pure.c so the rest of the codebase is unaware of the backend.
 */
#include <openblas/cblas.h>
#include "linalg.h"

void linalg_matvec(const double *A, int n, const double *x, double *y) {
    cblas_dgemv(CblasRowMajor, CblasNoTrans, n, n, 1.0, A, n, x, 1, 0.0, y, 1);
}

void linalg_ger(double *A, int n, double alpha, const double *x, const double *y) {
    cblas_dger(CblasRowMajor, n, n, alpha, x, 1, y, 1, A, n);
}

void linalg_axpy(int n, double alpha, const double *x, double *y) {
    cblas_daxpy(n, alpha, x, 1, y, 1);
}

double linalg_dot(int n, const double *x, const double *y) {
    return cblas_ddot(n, x, 1, y, 1);
}

void linalg_copy(int n, const double *x, double *y) {
    cblas_dcopy(n, x, 1, y, 1);
}

void linalg_scal(int n, double alpha, double *x) {
    cblas_dscal(n, alpha, x, 1);
}

void linalg_add_diag(double *A, int n, double d) {
    for (int i = 0; i < n; i++) A[(long)i * n + i] += d;
}
