#pragma once
#include "objective.hpp"
#include <vector>
#include <cblas.h>

class QuadraticFunction : public ObjectiveFunction {
public:
    // A row-major (n*n), b size n
    QuadraticFunction(int n,
                      std::vector<double> A_row_major,
                      std::vector<double> b);

    double value(const std::vector<double>& x) const override;
    void gradient(const std::vector<double>& x,
                  std::vector<double>& grad) const override;
    int dimension() const override { return n_; }

private:
    int n_;
    std::vector<double> A_;
    std::vector<double> b_;

    void matvec(const std::vector<double>& x,
                std::vector<double>& Ax) const;
};
