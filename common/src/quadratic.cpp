#include "quadratic.hpp"
#include <stdexcept>

QuadraticFunction::QuadraticFunction(int n,
                                     std::vector<double> A_row_major,
                                     std::vector<double> b)
    : n_(n), A_(std::move(A_row_major)), b_(std::move(b)) {
    if ((int)A_.size() != n_ * n_) throw std::runtime_error("A size != n*n");
    if ((int)b_.size() != n_) throw std::runtime_error("b size != n");
}

void QuadraticFunction::matvec(const std::vector<double>& x,
                               std::vector<double>& Ax) const {
    Ax.assign(n_, 0.0);
    cblas_dgemv(CblasRowMajor, CblasNoTrans,
                n_, n_,
                1.0, A_.data(), n_,
                x.data(), 1,
                0.0, Ax.data(), 1);
}

double QuadraticFunction::value(const std::vector<double>& x) const {
    std::vector<double> Ax;
    matvec(x, Ax);
    double xtAx = cblas_ddot(n_, x.data(), 1, Ax.data(), 1);
    double btx  = cblas_ddot(n_, b_.data(), 1, x.data(), 1);
    return 0.5 * xtAx - btx;
}

void QuadraticFunction::gradient(const std::vector<double>& x,
                                 std::vector<double>& grad) const {
    std::vector<double> Ax;
    matvec(x, Ax);
    grad.resize(n_);
    for (int i = 0; i < n_; ++i)
        grad[i] = Ax[i] - b_[i];
}
