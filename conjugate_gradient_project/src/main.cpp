#include "quadratic.hpp"
#include <cblas.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

static std::vector<double> A_from_3x3() {
    return { 4,1,2,
             1,3,1,
             2,1,5 };
}

static void matvec_rowmajor(int n, const std::vector<double>& A,
                            const std::vector<double>& x, std::vector<double>& Ax) {
    Ax.assign(n, 0.0);
    cblas_dgemv(CblasRowMajor, CblasNoTrans, n, n, 1.0,
                A.data(), n, x.data(), 1, 0.0, Ax.data(), 1);
}

int main() {
    int n = 3;
    auto A = A_from_3x3();
    std::vector<double> b = {1,2,3};

    // Para logging de f(x) usamos QuadraticFunction (misma f del repo)
    QuadraticFunction f(n, A, b);

    int max_iters = 1000;
    double tol = 1e-10;

    std::vector<double> x(n, 0.0);
    std::vector<double> Ax, r(n), p(n), Ap;

    // r0 = b - A x0
    matvec_rowmajor(n, A, x, Ax);
    for (int i = 0; i < n; ++i) r[i] = b[i] - Ax[i];
    p = r;

    double rsold = cblas_ddot(n, r.data(), 1, r.data(), 1);

    std::ofstream csv("cg.csv");
    csv << "iter,f,grad_norm,time_ms\n";

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int k = 1; k <= max_iters; ++k) {
        matvec_rowmajor(n, A, p, Ap);
        double denom = cblas_ddot(n, p.data(), 1, Ap.data(), 1);

        if (std::abs(denom) < 1e-30) {
            std::cout << "Break: division by ~0 (p^T A p)\n";
            break;
        }

        double alpha = rsold / denom;

        // x = x + alpha p
        cblas_daxpy(n, alpha, p.data(), 1, x.data(), 1);

        // r = r - alpha Ap
        cblas_daxpy(n, -alpha, Ap.data(), 1, r.data(), 1);

        double rsnew = cblas_ddot(n, r.data(), 1, r.data(), 1);
        double rnorm = std::sqrt(rsnew);

        double fx = f.value(x);
        auto now = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(now - t0).count();
        csv << k << "," << fx << "," << rnorm << "," << elapsed_ms << "\n";

        if (rnorm < tol) {
            std::cout << "CG converged at iter " << k << " with ||r||=" << rnorm << "\n";
            break;
        }

        double beta = rsnew / rsold;

        // p = r + beta p
        for (int i = 0; i < n; ++i) p[i] = r[i] + beta * p[i];

        rsold = rsnew;
    }

    csv.close();

    std::cout << "Solucion CG: ";
    for (double v : x) std::cout << v << " ";
    std::cout << "\n";
    return 0;
}
