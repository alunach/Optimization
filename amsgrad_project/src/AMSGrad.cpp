#include "AMSGrad.hpp"
#include <cblas.h>
#include <cmath>
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>

AMSGrad::AMSGrad(const ObjectiveFunction& f, AMSGradConfig cfg)
    : f_(f), n_(f.dimension()), cfg_(cfg), x_(n_, 0.0) {}

void AMSGrad::set_initial_x(const std::vector<double>& x0) {
    if ((int)x0.size() != n_) throw std::runtime_error("x0 size != dimension");
    x_ = x0;
}

void AMSGrad::optimize(const std::string& csv_path) {
    std::ofstream csv;
    if (!csv_path.empty()) {
        csv.open(csv_path);
        csv << "iter,f,grad_norm,time_ms\n";
    }

    std::vector<double> g(n_, 0.0);
    std::vector<double> m(n_, 0.0);
    std::vector<double> v(n_, 0.0);
    std::vector<double> vhat(n_, 0.0);

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int t = 1; t <= cfg_.max_iters; ++t) {
        f_.gradient(x_, g);

        for (int i = 0; i < n_; ++i) {
            m[i] = cfg_.beta1 * m[i] + (1.0 - cfg_.beta1) * g[i];
            v[i] = cfg_.beta2 * v[i] + (1.0 - cfg_.beta2) * (g[i] * g[i]);
            vhat[i] = std::max(vhat[i], v[i]); // AMSGrad: vhat_t = max(vhat_{t-1}, v_t)
        }

        // Bias correction
        double b1t = 1.0 - std::pow(cfg_.beta1, (double)t);
        double b2t = 1.0 - std::pow(cfg_.beta2, (double)t);

        for (int i = 0; i < n_; ++i) {
            double mhat = m[i] / b1t;
            double vhat_corr = vhat[i] / b2t; // variante común
            x_[i] -= cfg_.alpha * (mhat / (std::sqrt(vhat_corr) + cfg_.eps));
        }

        double gnorm = cblas_dnrm2(n_, g.data(), 1);
        double fx = f_.value(x_);

        auto now = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(now - t0).count();

        if (csv.is_open()) csv << t << "," << fx << "," << gnorm << "," << elapsed_ms << "\n";
        if (cfg_.verbose && (t == 1 || t % cfg_.log_every == 0)) {
            std::cout << "[iter " << t << "] f=" << fx << " ||g||=" << gnorm
                      << " time_ms=" << elapsed_ms << "\n";
        }

        if (gnorm < cfg_.tol_grad) {
            std::cout << "Converged at iter " << t << "\n";
            break;
        }
    }

    if (csv.is_open()) csv.close();
}
