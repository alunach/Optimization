#include "Adam.h"
#include "wine_quadratic.h"
#include <cblas.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>


int main() {
    try {
        const double lambda = 1e-3;     // ridge pequeño para SPD
        const bool normalize = true;    // MUY recomendado para GD

        auto ab = build_quadratic_from_csv("../data/wine.csv", lambda, normalize, 1, 1);
        int n = ab.n;
        auto A = ab.A;
        auto b = ab.b;

        // Adam params (sensibles para este setup)
        const double alpha = 0.01;
        const double beta1 = 0.9;
        const double beta2 = 0.999;
        const double eps = 1e-8;

        const int max_iters = 50000;
        const double tol_grad = 1e-6;

        Adam opt(n, std::move(A), std::move(b), alpha, beta1, beta2, eps, max_iters, tol_grad);
        auto res = opt.optimize_to_csv("../data_output/adam.csv");

        std::cout << "OK: adam.csv generado\n";
        std::cout << "iters=" << res.iters
                  << " final_f=" << res.final_f
                  << " final_grad_norm=" << res.final_grad_norm
                  << " time_ms=" << res.total_time_ms << "\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
