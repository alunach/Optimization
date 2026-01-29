#ifndef GRADIENT_DESCENT_H
#define GRADIENT_DESCENT_H

#include <vector>
#include <iostream>
#include <lapacke.h>
#include <cblas.h>
#include <chrono>

class GradientDescent {
private:
    std::vector<std::vector<double>> A; // Matriz de coeficientes (nxn)
    std::vector<double> b; // Vector de términos independientes (n)
    std::vector<double> x; // Vector de solución (n)
    double alpha; // Tasa de aprendizaje 0.01
    int max_iters; // Número máximo de iteraciones
    double tolerance; // Tolerancia para convergencia

public:
    GradientDescent(const std::vector<std::vector<double>>& A, const std::vector<double>& b,
                    double alpha, int max_iters, double tolerance);

    void optimize();
    void printSolution();
};

#endif
