#include "GradientDescent.h"

// Constructor: inicializa valores
GradientDescent::GradientDescent(const std::vector<std::vector<double>>& A, 
                                 const std::vector<double>& b, double alpha, 
                                 int max_iters, double tolerance)
    : A(A), b(b), alpha(alpha), max_iters(max_iters), tolerance(tolerance) {
    x.assign(b.size(), 0.0); // Inicializar x en 0
}

// Método de optimización con Descenso de Gradiente
void GradientDescent::optimize() {
    int n = A.size();
    std::vector<double> gradient(n, 0.0);
    std::vector<double> Ax(n, 0.0);

    auto start_time = std::chrono::high_resolution_clock::now(); // Iniciar medición de tiempo

    for (int iter = 0; iter < max_iters; iter++) {
        // Calcular Ax usando BLAS: Ax = A * x
        cblas_dgemv(CblasRowMajor, CblasNoTrans, n, n, 1.0, &A[0][0], n, &x[0], 1, 0.0, &Ax[0], 1);

        // Calcular el gradiente: grad = A * x - b
        for (int i = 0; i < n; i++)
            gradient[i] = Ax[i] - b[i];

        // Calcular la norma del gradiente para verificar convergencia
        double grad_norm = cblas_dnrm2(n, &gradient[0], 1);
        if (grad_norm < tolerance) {
            std::cout << "Convergencia alcanzada en iteración " << iter << std::endl;
            break;
        }

        // Actualizar x: x = x - alpha * gradient
        cblas_daxpy(n, -alpha, &gradient[0], 1, &x[0], 1);
    }

    auto end_time = std::chrono::high_resolution_clock::now(); // Finalizar medición de tiempo
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "Tiempo de ejecución: " << elapsed.count() << " segundos\n";
}

// Método para imprimir la solución final
void GradientDescent::printSolution() {
    std::cout << "Solución óptima x: ";
    for (double val : x) std::cout << val << " ";
    std::cout << std::endl;
}
