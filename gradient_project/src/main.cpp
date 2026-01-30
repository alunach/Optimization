#include "GradientDescent.h"
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

        // hiperparámetros: GD es sensible, empezamos conservador
        const double alpha = 0.01;
        const int max_iters = 50000;
        const double tol_grad = 1e-6;

        GradientDescent gd(n, std::move(A), std::move(b), alpha, max_iters, tol_grad);
        auto res = gd.optimize_to_csv("../data_output/gradient.csv");

        std::cout << "OK: gradient.csv generado\n";
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
