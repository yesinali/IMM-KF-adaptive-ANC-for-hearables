/* Pure C backend: no external math library; relies only on the compiler's
 * autovectorizer (-O3 -march=native typically hits AVX2 on modern x86).
 */
#include "linalg.h"

void linalg_matvec(const double *A, int n, const double *x, double *y) {
    for (int i = 0; i < n; i++) {
        double s = 0.0;
        const double *Ai = A + (long)i * n;
        for (int j = 0; j < n; j++) s += Ai[j] * x[j];
        y[i] = s;
    }
}

void linalg_ger(double *A, int n, double alpha, const double *x, const double *y) {
    for (int i = 0; i < n; i++) {
        double xi = alpha * x[i];
        double *Ai = A + (long)i * n;
        for (int j = 0; j < n; j++) Ai[j] += xi * y[j];
    }
}

void linalg_axpy(int n, double alpha, const double *x, double *y) {
    for (int i = 0; i < n; i++) y[i] += alpha * x[i];
}

double linalg_dot(int n, const double *x, const double *y) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += x[i] * y[i];
    return s;
}

void linalg_copy(int n, const double *x, double *y) {
    for (int i = 0; i < n; i++) y[i] = x[i];
}

void linalg_scal(int n, double alpha, double *x) {
    for (int i = 0; i < n; i++) x[i] *= alpha;
}

void linalg_add_diag(double *A, int n, double d) {
    for (int i = 0; i < n; i++) A[(long)i * n + i] += d;
}
