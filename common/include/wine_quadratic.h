// common/include/wine_quadratic.h
#pragma once
#include <string>
#include <vector>

struct QuadraticAB {
    int n = 0;                         // dimension = #features (d)
    std::vector<double> A;             // row-major (n*n)
    std::vector<double> b;             // (n)
};

// Construye A = X^T X + lambda I, b = X^T y
// - path: ruta al CSV (ej: "../data/wine.csv")
// - lambda: regularización ridge (0.0 = sin ridge)
// - normalize: estandariza features (z-score) para estabilidad de GD/NAG
// - y_mode: 0 = usar última columna como y (regresión)
//          1 = binario one-vs-rest: y=1 si label==positive_class, si no 0
QuadraticAB build_quadratic_from_csv(
    const std::string& path,
    double lambda = 1e-3,
    bool normalize = true,
    int y_mode = 0,
    int positive_class = 1
);
