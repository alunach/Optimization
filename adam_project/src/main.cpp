#include "Adam.h"
#include "wine_quadratic.h"
#include <iostream>

static std::vector<std::vector<double>> to2D(int n, const std::vector<double>& Arow) {
    std::vector<std::vector<double>> A(n, std::vector<double>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            A[i][j] = Arow[(size_t)i*n + j];
    return A;
}

int main() {
    // Definir la matriz A (3x3) y el vector b (3)
    //std::vector<std::vector<double>> A = {{4, 1, 2}, {1, 3, 1}, {2, 1, 5}};
    //std::vector<double> b = {1, 2, 3};
    auto ab = build_quadratic_from_csv("../../data/wine.csv", 1e-3, true, 1, 1);
    int n = ab.n;
    auto A = to2D(n, ab.A);
    auto b = ab.b;

    double alpha = 0.01;  // Tasa de aprendizaje
    double beta1 = 0.9;
    double beta2 = 0.999;
    double epsilon = 1e-8;
    int max_iters = 10000; // Número máximo de iteraciones
    double tolerance = 1e-6; // Criterio de convergencia

    // Crear objeto Adam
    Adam adam(A, b, alpha, beta1, beta2, epsilon, max_iters, tolerance);

    // Ejecutar optimización
    adam.optimize();

    // Imprimir la solución encontrada
    adam.printSolution();

    return 0;
}
