#include "Adam.h"
#include <iostream>

int main() {
    // Definir la matriz A (3x3) y el vector b (3)
    std::vector<std::vector<double>> A = {{4, 1, 2}, {1, 3, 1}, {2, 1, 5}};
    std::vector<double> b = {1, 2, 3};

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
