#ifndef ADAM_H
#define ADAM_H

#include <vector>
#include <iostream>
#include <lapacke.h>
#include <cblas.h>
#include <chrono>
#include <cmath>

class Adam {
private:
    std::vector<std::vector<double>> A; // Matriz de coeficientes (nxn)
    std::vector<double> b; // Vector de términos independientes (n)
    std::vector<double> x; // Vector de solución (n)
    double alpha; // Tasa de aprendizaje
    int max_iters; // Número máximo de iteraciones
    double tolerance; // Tolerancia para convergencia
    double beta1, beta2, epsilon; // Parámetros de Adam

public:
    Adam(const std::vector<std::vector<double>>& A, const std::vector<double>& b,
         double alpha, double beta1, double beta2, double epsilon, int max_iters, double tolerance);

    void optimize();
    void printSolution();
};

#endif
