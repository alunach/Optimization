#include "GradientDescent.h"
#include <iostream>

int main() {
    // Definir la matriz A (3x3) y el vector b (3)
    std::vector<std::vector<double>> A = {{4, 1, 2}, {1, 3, 1}, {2, 1, 5}};
    std::vector<double> b = {1, 2, 3};

    double alpha = 0.01;  // Tasa de aprendizaje
    int max_iters = 1000; // Número máximo de iteraciones
    double tolerance = 1e-6; // Criterio de convergencia

    // Crear objeto GradientDescent
    GradientDescent gd(A, b, alpha, max_iters, tolerance);

    // Ejecutar optimización
    gd.optimize();

    // Imprimir la solución encontrada
    gd.printSolution();

    return 0;
}
