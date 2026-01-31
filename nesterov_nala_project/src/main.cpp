#include "nala.hpp"
#include "quadratic.hpp"
#include "wine_quadratic.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>

static QuadraticCase parse_case(int argc, char** argv) {
    int c = 1;
    if (argc >= 2) c = std::atoi(argv[1]);
    switch (c) {
        case 0: return QuadraticCase::ConvexPSD;
        case 1: return QuadraticCase::StronglyConvexSPD;
        case 2: return QuadraticCase::IllConditionedSPD;
        case 3: return QuadraticCase::NonConvexIndefinite;
        default: return QuadraticCase::StronglyConvexSPD;
    }
}
static const char* case_name(QuadraticCase qc) {
    switch (qc) {
        case QuadraticCase::ConvexPSD: return "convex";
        case QuadraticCase::StronglyConvexSPD: return "strong";
        case QuadraticCase::IllConditionedSPD: return "illcond";
        case QuadraticCase::NonConvexIndefinite: return "nonconvex";
        default: return "strong";
    }
}

int main(int argc, char** argv) {
    const double lambda = 1e-3;
    const bool normalize = true;
    const int y_mode = 1;
    const int positive_class = 1;

    const QuadraticCase qc = parse_case(argc, argv);

    auto ab = build_quadratic_case_from_csv(
        "../data/wine.csv",
        qc,
        lambda,
        normalize,
        y_mode,
        positive_class,
        /*illcond_ratio=*/1e8,
        /*nonconvex_delta=*/1.0
    );

    QuadraticFunction f(ab.n, ab.A, ab.b);

    NALAConfig cfg;
    cfg.outer_iters = 2000;
    cfg.k_sync = 5;

    // Adam interno
    cfg.lr_fast = 1e-3;
    cfg.beta1 = 0.9;
    cfg.beta2 = 0.999;
    cfg.eps = 1e-8;

    // Look-ahead (slow)
    cfg.mu = -0.5;
    cfg.alpha_slow = 1e-3;

    cfg.tol_grad = 1e-6;

    NALA opt(f, cfg);
    opt.set_initial_phi(std::vector<double>(ab.n, 0.0));

    const std::string out_csv = std::string("../data_output/nala_") + case_name(qc) + ".csv";
    opt.optimize(out_csv);

    const auto& sol = opt.solution();
    std::cout << "OK: " << out_csv << " generado\n";
    std::cout << "Solucion NALA: ";
    for (double v : sol) std::cout << v << " ";
    std::cout << "\n";
    return 0;
}
