// common/include/wine_quadratic.h
#pragma once
#include <string>
#include <vector>

struct QuadraticAB {
    int n = 0;                 // dimensión (d)
    std::vector<double> A;     // row-major (n*n)
    std::vector<double> b;     // (n)
};

// Construye A0 = X^T X + lambda I, b = X^T y desde wine.csv
// IMPORTANTE: wine.csv tiene clase en la PRIMERA columna.
// - path: "../data/wine.csv"
// - normalize: z-score por columna de X (recomendado)
// - y_mode: 0 = regresión (y = clase como double), 1 = one-vs-rest binario
// - positive_class: etiqueta positiva (1,2,3) si y_mode=1
QuadraticAB build_quadratic_from_wine_csv(
    const std::string& path,
    double lambda,
    bool normalize,
    int y_mode,
    int positive_class
);

enum class QuadraticCase {
    ConvexPSD = 0,
    StronglyConvexSPD = 1,
    IllConditionedSPD = 2,
    NonConvexIndefinite = 3
};

// Construye (A,b) desde wine.csv y transforma SOLO A para forzar el caso
QuadraticAB build_quadratic_case_from_wine_csv(
    const std::string& path,
    QuadraticCase which,
    double lambda,
    bool normalize,
    int y_mode,
    int positive_class,
    double illcond_ratio = 1e8,
    double nonconvex_delta = 1.0
);
