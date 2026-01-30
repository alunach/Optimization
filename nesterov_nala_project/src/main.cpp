#include "nala.hpp"
#include "quadratic.hpp"
#include "wine_quadratic.h"
#include <iostream>
#include <vector>
#include <iostream>
#include <cmath>

// helper caso: mal condicionada
/*static std::vector<double> diagA(int n, double a0, double a1) {
    std::vector<double> A(n * n, 0.0);
    for (int i = 0; i < n; ++i) {
        double t = (double)i / (n - 1);
        double v = std::exp(std::log(a0)*(1-t) + std::log(a1)*t);
        A[i * n + i] = v;
    }
    return A;
}*/

/*
static std::vector<double> A_from_3x3() {
    // Row-major 3x3:
    // [4 1 2
    //  1 3 1
    //  2 1 5]
    return {
        4,1,2,
        1,3,1,
        2,1,5
    };
}
*/

int main() {
    //int n = 3;
    //auto A = A_from_3x3();
    //std::vector<double> b = {1,2,3};
    auto ab = build_quadratic_from_csv("../data/wine.csv", 1e-3, true, 1, 1);
    int n = ab.n;
    auto A = ab.A;
    auto b = ab.b;


    QuadraticFunction f(n, A, b);

    NALAConfig cfg;
    cfg.outer_iters = 2000;
    cfg.k_sync = 5;

    // Parámetros típicos (ajústalos en experimentos)
    cfg.lr_fast = 1e-3;     // Adam interno
    cfg.beta1 = 0.9;
    cfg.beta2 = 0.999;
    cfg.eps = 1e-8;

    cfg.mu = -0.5;          // look-ahead
    cfg.alpha_slow = 1e-3;  // paso lento
    cfg.tol_grad = 1e-6;

    NALA nala(f, cfg);
    nala.set_initial_phi(std::vector<double>(n, 0.0));
    nala.optimize("../data_output/nala.csv");

    const auto& sol = nala.solution();
    std::cout << "Solucion NALA: ";
    for (double v : sol) std::cout << v << " ";
    std::cout << "\n";
    return 0;

    // Caso: mal condicionada
    /*
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
    std::cout << "\n";*/
}
