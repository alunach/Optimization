#include "wine_quadratic.h"
#include "quadratic.hpp"
#include <cblas.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>

static QuadraticCase parse_case(int argc, char** argv) {
    int c = 1;
    if (argc >= 2) c = std::atoi(argv[1]);
    switch (c) {
        case 0: return QuadraticCase::ConvexPSD;
        case 1: return QuadraticCase::StronglyConvexSPD;
        case 2: return QuadraticCase::IllConditionedSPD;
        case 3: return QuadraticCase::NonConvexIndefinite;
        default: return QuadraticCase::StronglyConvexSPD;
    }
}
static const char* case_name(QuadraticCase qc) {
    switch (qc) {
        case QuadraticCase::ConvexPSD: return "convex";
        case QuadraticCase::StronglyConvexSPD: return "strong";
        case QuadraticCase::IllConditionedSPD: return "illcond";
        case QuadraticCase::NonConvexIndefinite: return "nonconvex";
        default: return "strong";
    }
}

static void matvec(int n, const std::vector<double>& A, const std::vector<double>& x, std::vector<double>& Ax) {
    Ax.assign(n, 0.0);
    cblas_dgemv(CblasRowMajor, CblasNoTrans, n, n, 1.0,
                A.data(), n, x.data(), 1, 0.0, Ax.data(), 1);
}

int main(int argc, char** argv) {
    const double lambda = 1e-3;
    const bool normalize = true;
    const int y_mode = 1;
    const int positive_class = 1;

    const QuadraticCase qc = parse_case(argc, argv);

    auto ab = build_quadratic_case_from_csv(
        "../data/wine.csv",
        qc,
        lambda,
        normalize,
        y_mode,
        positive_class,
        /*illcond_ratio=*/1e8,
        /*nonconvex_delta=*/1.0
    );

    const int n = ab.n;

    QuadraticFunction f(n, ab.A, ab.b);

    std::vector<double> x(n, 0.0), Ax, r(n), p(n), Ap;

    // r0 = b - A x0
    matvec(n, ab.A, x, Ax);
    for (int i = 0; i < n; ++i) r[i] = ab.b[i] - Ax[i];
    p = r;

    double rsold = cblas_ddot(n, r.data(), 1, r.data(), 1);

    const std::string out_csv = std::string("../data_output/cg_") + case_name(qc) + ".csv";
    std::ofstream csv(out_csv);
    csv << "iter,f,grad_norm,time_ms\n";

    auto t0 = std::chrono::high_resolution_clock::now();

    const int max_iters = 5000;
    const double tol = 1e-6;

    for (int k = 1; k <= max_iters; ++k) {
        matvec(n, ab.A, p, Ap);
        double denom = cblas_ddot(n, p.data(), 1, Ap.data(), 1);
        if (std::abs(denom) < 1e-30) break;

        double alpha = rsold / denom;

        cblas_daxpy(n, alpha, p.data(), 1, x.data(), 1);
        cblas_daxpy(n, -alpha, Ap.data(), 1, r.data(), 1);

        double rsnew = cblas_ddot(n, r.data(), 1, r.data(), 1);
        double rnorm = std::sqrt(rsnew);

        double fx = f.value(x);
        auto now = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(now - t0).count();
        csv << k << "," << fx << "," << rnorm << "," << ms << "\n";

        if (rnorm < tol) {
            std::cout << "CG converged at iter " << k << " with ||r||=" << rnorm << "\n";
            break;
        }

        double beta = rsnew / rsold;
        for (int i = 0; i < n; ++i) p[i] = r[i] + beta * p[i];
        rsold = rsnew;
    }

    std::cout << "OK: " << out_csv << " generado (dim=" << n << ")\n";
    return 0;
}
