#include "nala.hpp"
#include "quadratic.hpp"
#include "wine_quadratic.h"
#include <iostream>
#include <vector>

int main() {
    auto ab = build_quadratic_from_csv("../../data/wine.csv", /*lambda=*/1e-3, /*normalize=*/true,
                                       /*y_mode=*/1, /*positive_class=*/1);

    QuadraticFunction f(ab.n, ab.A, ab.b);

    NALAConfig cfg;
    cfg.outer_iters = 2000;
    cfg.k_sync = 5;
    cfg.lr_fast = 1e-3;
    cfg.beta1 = 0.9;
    cfg.beta2 = 0.999;
    cfg.eps = 1e-8;
    cfg.mu = -0.5;
    cfg.alpha_slow = 1e-3;
    cfg.tol_grad = 1e-6;

    NALA nala(f, cfg);
    nala.set_initial_phi(std::vector<double>(ab.n, 0.0));
    nala.optimize("nala.csv");

    std::cout << "NALA ok. dim=" << ab.n << "\n";
    return 0;
}

/*
#include "nala.hpp"

#include <cmath>
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>

NALA::NALA(const ObjectiveFunction& f, NALAConfig cfg)
    : f_(f), n_(f.dimension()), phi_(n_, 0.0), cfg_(cfg) {}

void NALA::set_initial_phi(const std::vector<double>& init) {
    if ((int)init.size() != n_)
        throw std::runtime_error("init size != dimension");
    phi_ = init;
}

void NALA::optimize(const std::string& csv_path) {
    std::ofstream csv;
    if (!csv_path.empty()) {
        csv.open(csv_path);
        csv << "iter,f,grad_norm,time_ms\n";
    }

    std::vector<double> theta(n_), y(n_), g(n_);
    std::vector<double> m(n_, 0.0), v(n_, 0.0);
    long long adam_step = 0;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int t = 1; t <= cfg_.outer_iters; ++t) {
        theta = phi_;

        // Inner Adam
        for (int i = 0; i < cfg_.k_sync; ++i) {
            ++adam_step;
            f_.gradient(theta, g);

            for (int j = 0; j < n_; ++j) {
                m[j] = cfg_.beta1 * m[j] + (1.0 - cfg_.beta1) * g[j];
                v[j] = cfg_.beta2 * v[j] + (1.0 - cfg_.beta2) * g[j] * g[j];
            }

            double b1t = 1.0 - std::pow(cfg_.beta1, (double)adam_step);
            double b2t = 1.0 - std::pow(cfg_.beta2, (double)adam_step);

            for (int j = 0; j < n_; ++j) {
                double mhat = m[j] / b1t;
                double vhat = v[j] / b2t;
                theta[j] -= cfg_.lr_fast * (mhat / (std::sqrt(vhat) + cfg_.eps));
            }
        }

        // Extrapolation
        for (int j = 0; j < n_; ++j)
            y[j] = (1.0 + cfg_.mu) * theta[j] - cfg_.mu * phi_[j];

        // Slow update
        f_.gradient(y, g);
        for (int j = 0; j < n_; ++j)
            phi_[j] = y[j] - cfg_.alpha_slow * g[j];

        // Logging
        f_.gradient(phi_, g);
        double gnorm = cblas_dnrm2(n_, g.data(), 1);
        double fx = f_.value(phi_);

        auto now = std::chrono::high_resolution_clock::now();
        double elapsed_ms =
            std::chrono::duration<double, std::milli>(now - t0).count();

        if (csv.is_open())
            csv << t << "," << fx << "," << gnorm << "," << elapsed_ms << "\n";

        if (cfg_.verbose && (t == 1 || t % cfg_.log_every == 0)) {
            std::cout << "[outer " << t << "] f=" << fx
                      << " ||g||=" << gnorm
                      << " time_ms=" << elapsed_ms << "\n";
        }

        if (gnorm < cfg_.tol_grad) {
            std::cout << "Converged at outer iter " << t << "\n";
            break;
        }
    }

    if (csv.is_open()) csv.close();
}

*/