#include "include/nala.hpp"
#include "include/quadratic.hpp"
#include <iostream>
#include <cmath>

// helper
static std::vector<double> diagA(int n, double a0, double a1) {
    std::vector<double> A(n * n, 0.0);
    for (int i = 0; i < n; ++i) {
        double t = (double)i / (n - 1);
        double v = std::exp(std::log(a0)*(1-t) + std::log(a1)*t);
        A[i * n + i] = v;
    }
    return A;
}

int main() {
    int n = 10;

    auto A = diagA(n, 1e-3, 1e3); // mal condicionada
    std::vector<double> b(n, 1.0);

    QuadraticFunction f(n, A, b);

    NALAConfig cfg;
    cfg.outer_iters = 1500;
    cfg.k_sync = 5;
    cfg.mu = -0.5;
    cfg.alpha_slow = 1e-3;

    NALA nala(f, cfg);

    nala.set_initial_phi(std::vector<double>(n, 0.0));
    nala.optimize("nala.csv");

    const auto& sol = nala.solution();
    std::cout << "phi*[0..4]: ";
    for (int i = 0; i < std::min(5, n); ++i)
        std::cout << sol[i] << " ";
    std::cout << "\n";
}
