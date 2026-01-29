#ifndef MINIBATCH_SGD_H
#define MINIBATCH_SGD_H

#include <vector>
#include <iostream>
#include <lapacke.h>
#include <cblas.h>
#include <chrono>
#include <random>

class MiniBatchSGD {
private:
    std::vector<std::vector<double>> A; // Matriz de coeficientes (nxn)
    std::vector<double> b; // Vector de términos independientes (n)
    std::vector<double> x; // Vector de solución (n)
    double alpha; // Tasa de aprendizaje
    int max_iters; // Número máximo de iteraciones
    double tolerance; // Tolerancia para convergencia
    int batch_size; // Tamaño del mini-batch

public:
    MiniBatchSGD(const std::vector<std::vector<double>>& A, const std::vector<double>& b,
                 double alpha, int max_iters, double tolerance, int batch_size);

    void optimize();
    void printSolution();
};

#endif
