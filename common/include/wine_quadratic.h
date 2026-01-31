// common/include/wine_quadratic.h
#pragma once
#include <string>
#include <vector>

struct QuadraticAB {
    int n = 0;                 // dimensión (d)
    std::vector<double> A;     // row-major (n*n)
    std::vector<double> b;     // (n)
};

// Construye A0 = X^T X + lambda I, b = X^T y a partir de wine.csv
// - path: ruta al CSV (ej: "../data/wine.csv")
// - normalize: z-score por columna para estabilidad numérica (recomendado)
// - y_mode: 0=usar la última columna como y (regresión), 1=one-vs-rest binario
// - positive_class: etiqueta positiva si y_mode=1
QuadraticAB build_quadratic_from_csv(
    const std::string& path,
    double lambda,
    bool normalize,
    int y_mode,
    int positive_class
);

// Casos de la Parte A (solo se MODIFICA A; b se mantiene)
enum class QuadraticCase {
    ConvexPSD = 0,           // convexa (semidefinida positiva; puede no ser fuertemente convexa)
    StronglyConvexSPD = 1,   // fuertemente convexa (definida positiva)
    IllConditionedSPD = 2,   // SPD pero mal condicionada
    NonConvexIndefinite = 3  // no convexa (indefinida)
};

// Construye (A,b) desde wine.csv y transforma SOLO la matriz A para forzar el caso.
// - ConvexPSD: A = X^T X (PSD), opcionalmente con ridge=0.
// - StronglyConvexSPD: A = X^T X + lambda I  (lambda>0 asegura SPD).
// - IllConditionedSPD: A = Q diag(s) Q^T (misma dimensión) con cond(A)=illcond_ratio.
// - NonConvexIndefinite: A = A_spd - delta * v v^T con v autovector del menor autovalor (fuerza autovalor negativo).
QuadraticAB build_quadratic_case_from_csv(
    const std::string& path,
    QuadraticCase which,
    double lambda,
    bool normalize,
    int y_mode,
    int positive_class,
    double illcond_ratio = 1e8,
    double nonconvex_delta = 1.0
);
