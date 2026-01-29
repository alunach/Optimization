#ifndef AUTO_DIFF_H
#define AUTO_DIFF_H

#include <vector>
#include <iostream>
#include <cblas.h>  // Para operaciones BLAS

class AutoDiff {
private:
    std::vector<std::vector<double>> W;  // Matriz de coeficientes
    std::vector<double> x;               // Vector de entrada
    std::vector<double> y;               // Resultado de f(Wx)
    std::vector<double> grad;            // Gradiente df/dx

public:
    AutoDiff(const std::vector<std::vector<double>>& W, const std::vector<double>& x);
    void forward_and_backward();
    void printResults();
};

#endif
