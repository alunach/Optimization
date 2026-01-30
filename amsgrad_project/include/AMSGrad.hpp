#pragma once
#include "objective.hpp"
#include <vector>
#include <string>

struct AMSGradConfig {
    double alpha = 1e-2;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double eps   = 1e-8;
    int max_iters = 10000;
    double tol_grad = 1e-6;

    bool verbose = true;
    int log_every = 50;
};

class AMSGrad {
public:
    AMSGrad(const ObjectiveFunction& f, AMSGradConfig cfg);

    void set_initial_x(const std::vector<double>& x0);
    void optimize(const std::string& csv_path = "");
    const std::vector<double>& solution() const { return x_; }

private:
    const ObjectiveFunction& f_;
    int n_;
    AMSGradConfig cfg_;

    std::vector<double> x_;
};
