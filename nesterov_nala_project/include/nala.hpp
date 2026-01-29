#pragma once
#include "objective.hpp"
#include <vector>
#include <string>

#include <cblas.h>

struct NALAConfig {
    int outer_iters = 2000;
    int k_sync = 5;

    double alpha_slow = 1e-3;
    double mu = -0.5;
    double tol_grad = 1e-6;

    // Adam (inner)
    double lr_fast = 1e-3;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double eps = 1e-8;

    bool verbose = true;
    int log_every = 50;
};

class NALA {
public:
    NALA(const ObjectiveFunction& f, NALAConfig cfg);

    void set_initial_phi(const std::vector<double>& init);
    void optimize(const std::string& csv_path = "");
    const std::vector<double>& solution() const { return phi_; }

private:
    const ObjectiveFunction& f_;
    int n_;
    std::vector<double> phi_;
    NALAConfig cfg_;
};
