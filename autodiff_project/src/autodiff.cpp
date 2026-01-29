#include "autodiff.h"

// Constructor
AutoDiff::AutoDiff(const std::vector<std::vector<double>>& W, const std::vector<double>& x) 
    : W(W), x(x), y(W.size(), 0.0), grad(x.size(), 0.0) {}

// Función de activación (f(z) = z^2)
double activation(double z) { return z * z; }

// Derivada de la función de activación (f'(z) = 2z)
double activation_derivative(double z) { return 2 * z; }

// Cálculo de salida y derivadas
void AutoDiff::forward_and_backward() {
    int rows = W.size();  // Filas de la matriz W
    int cols = W[0].size(); // Columnas de la matriz W

    std::vector<double> Wx(rows, 0.0);

    // Multiplicación Wx = W * x usando BLAS
    cblas_dgemv(CblasRowMajor, CblasNoTrans, rows, cols, 1.0, &W[0][0], cols, &x[0], 1, 0.0, &Wx[0], 1);

    // Aplicar función de activación y calcular gradientes
    for (int i = 0; i < rows; i++) {
        y[i] = activation(Wx[i]); // f(Wx)
        double dz_dx = activation_derivative(Wx[i]); // f'(Wx)
        
        // Aplicar regla de la cadena
        for (int j = 0; j < cols; j++) {
            grad[j] += dz_dx * W[i][j];
        }
    }
}

// Imprimir los resultados
void AutoDiff::printResults() {
    std::cout << "Salida y = f(Wx): ";
    for (double yi : y) std::cout << yi << " ";
    std::cout << std::endl;

    std::cout << "Gradiente df/dx: ";
    for (double gi : grad) std::cout << gi << " ";
    std::cout << std::endl;
}
