#include "AMSGrad.hpp"
#include "quadratic.hpp"
#include "wine_quadratic.h"
#include <iostream>
#include <vector>

int main() {
    auto ab = build_quadratic_from_csv("../data/wine.csv", 1e-3, true, 1, 1);

    QuadraticFunction f(ab.n, ab.A, ab.b);

    AMSGradConfig cfg;
    cfg.alpha = 0.01;
    cfg.max_iters = 20000;
    cfg.tol_grad = 1e-6;

    AMSGrad opt(f, cfg);
    opt.set_initial_x(std::vector<double>(ab.n, 0.0));
    opt.optimize("../data_output/amsgrad.csv");

    std::cout << "AMSGrad ok. dim=" << ab.n << "\n";
    return 0;
}

/*
#include "AMSGrad.hpp"
#include "quadratic.hpp"
#include <iostream>
#include <vector>

static std::vector<double> A_from_3x3() {
    return { 4,1,2,
             1,3,1,
             2,1,5 };
}

int main() {
    int n = 3;
    auto A = A_from_3x3();
    std::vector<double> b = {1,2,3};

    QuadraticFunction f(n, A, b);

    AMSGradConfig cfg;
    cfg.alpha = 0.01;
    cfg.max_iters = 10000;
    cfg.tol_grad = 1e-6;

    AMSGrad opt(f, cfg);
    opt.set_initial_x(std::vector<double>(n, 0.0));
    opt.optimize("amsgrad.csv");

    std::cout << "Solucion AMSGrad: ";
    for (double v : opt.solution()) std::cout << v << " ";
    std::cout << "\n";
    return 0;
}
*/