#include "Nesterov.h"

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
    try {
        const double lambda = 1e-3;
        const bool normalize = true;
        const int y_mode = 1;
        const int positive_class = 1;

        const QuadraticCase qc = parse_case(argc, argv);

        auto ab = build_quadratic_case_from_wine_csv(
            "../data/wine.csv",
            qc,
            lambda,
            normalize,
            y_mode,
            positive_class,
            /*illcond_ratio=*/1e8,
            /*nonconvex_delta=*/1.0
        );

        const int n = ab.n;

        // Hiperparámetros NAG (Nesterov)
        const double alpha = 0.01;
        const double momentum = 0.9;
        const int max_iters = 50000;
        const double tol_grad = 1e-6;

        Nesterov opt(n, std::move(ab.A), std::move(ab.b), alpha, momentum, max_iters, tol_grad);

        const std::string out_csv = std::string("../data_output/nesterov_") + case_name(qc) + ".csv";
        auto res = opt.optimize_to_csv(out_csv);

        std::cout << "OK: " << out_csv << " generado\n";
        std::cout << "iters=" << res.iters
                  << " final_f=" << res.final_f
                  << " final_grad_norm=" << res.final_grad_norm
                  << " time_ms=" << res.total_time_ms << "\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
