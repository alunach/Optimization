#include "GradientDescent.h"
#include <cblas.h>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>

GradientDescent::GradientDescent(int n,
                                 std::vector<double> A_row_major,
                                 std::vector<double> b,
                                 double alpha,
                                 int max_iters,
                                 double tol_grad)
    : n_(n),
      A_(std::move(A_row_major)),
      b_(std::move(b)),
      alpha_(alpha),
      max_iters_(max_iters),
      tol_grad_(tol_grad) {
    if (n_ <= 0) throw std::runtime_error("n invalido");
    if ((int)A_.size() != n_ * n_) throw std::runtime_error("A size != n*n");
    if ((int)b_.size() != n_) throw std::runtime_error("b size != n");
}

double GradientDescent::f_value(const std::vector<double>& x,
                                const std::vector<double>& Ax) const {
    // f(x) = 0.5 x^T(Ax) - b^T x
    const double xtAx = cblas_ddot(n_, x.data(), 1, Ax.data(), 1);
    const double btx  = cblas_ddot(n_, b_.data(), 1, x.data(), 1);
    return 0.5 * xtAx - btx;
}

GradientDescent::Result GradientDescent::optimize_to_csv(const std::string& csv_path,
                                                         const std::vector<double>& x0) {
    std::ofstream csv(csv_path);
    if (!csv.is_open()) throw std::runtime_error("No se pudo abrir CSV: " + csv_path);
    csv << "iter,f,grad_norm,time_ms\n";

    std::vector<double> x(n_, 0.0);
    if (!x0.empty()) {
        if ((int)x0.size() != n_) throw std::runtime_error("x0 size != n");
        x = x0;
    }

    std::vector<double> Ax(n_, 0.0);
    std::vector<double> grad(n_, 0.0);

    auto t0 = std::chrono::high_resolution_clock::now();

    Result res;
    for (int iter = 1; iter <= max_iters_; ++iter) {
        // Ax = A*x
        cblas_dgemv(CblasRowMajor, CblasNoTrans, n_, n_,
                    1.0, A_.data(), n_,
                    x.data(), 1,
                    0.0, Ax.data(), 1);

        // grad = Ax - b
        for (int i = 0; i < n_; ++i) grad[i] = Ax[i] - b_[i];

        const double grad_norm = cblas_dnrm2(n_, grad.data(), 1);
        const double f = f_value(x, Ax);

        auto now = std::chrono::high_resolution_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(now - t0).count();

        csv << iter << "," << std::setprecision(15) << f << "," << grad_norm << "," << ms << "\n";

        // stop
        if (grad_norm < tol_grad_) {
            res.iters = iter;
            res.final_f = f;
            res.final_grad_norm = grad_norm;
            break;
        }

        // x <- x - alpha * grad
        for (int i = 0; i < n_; ++i) x[i] -= alpha_ * grad[i];

        // si llega al final sin converger
        if (iter == max_iters_) {
            res.iters = iter;
            res.final_f = f;
            res.final_grad_norm = grad_norm;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    res.total_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    res.x = std::move(x);
    return res;
}
