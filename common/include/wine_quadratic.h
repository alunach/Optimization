#pragma once
#include <string>
#include <vector>

struct QuadraticAB {
    int n = 0;                         // dimension = #features (d)
    std::vector<double> A;             // row-major (n*n)
    std::vector<double> b;             // (n)
};

// Construye una cuadrática f(x)= 0.5 x^T A x - b^T x, donde:
//
// y_mode=0 (regresión):
//   - y = última columna
//   - X = todas las columnas excepto la última
//
// y_mode=1 (one-vs-rest para Wine):
//   - label = primera columna (Wine: 1/2/3)
//   - y = 1 si label == positive_class, si no 0
//   - X = columnas 2..p (todas excepto la primera)
//
// Luego:
//   A = X^T X + lambda I
//   b = X^T y
QuadraticAB build_quadratic_from_csv(
    const std::string& path,
    double lambda = 1e-3,
    bool normalize = true,
    int y_mode = 0,
    int positive_class = 1
);
