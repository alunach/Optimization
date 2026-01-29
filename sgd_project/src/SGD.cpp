#include "SGD.h"
#include <random>

// Constructor: inicializa valores
SGD::SGD(const std::vector<std::vector<double>>& A, 
         const std::vector<double>& b, double alpha, 
         int max_iters, double tolerance)
    : A(A), b(b), alpha(alpha), max_iters(max_iters), tolerance(tolerance) {
    x.assign(b.size(), 0.0); // Inicializar x en 0
}

// Método de optimización con Stochastic Gradient Descent
void SGD::optimize() {
    int n = A.size();
    std::vector<double> Ax(n, 0.0);
    std::vector<double> gradient(n, 0.0);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, n - 1);

    auto start_time = std::chrono::high_resolution_clock::now(); // Iniciar medición de tiempo

    for (int iter = 0; iter < max_iters; iter++) {
        // Elegir un índice aleatorio
        int i = dist(gen);

        // Calcular Ax usando BLAS (solo la fila i): Ax[i] = A[i] * x
        Ax[i] = cblas_ddot(n, &A[i][0], 1, &x[0], 1);

        // Calcular el gradiente para la muestra i: grad_i = A[i] * x - b[i]
        gradient[i] = Ax[i] - b[i];

        // Actualizar x usando la muestra seleccionada
        cblas_daxpy(n, -alpha * gradient[i], &A[i][0], 1, &x[0], 1);

        // Calcular la norma del gradiente para verificar convergencia
        double grad_norm = cblas_dnrm2(n, &gradient[0], 1);
        if (grad_norm < tolerance) {
            std::cout << "Convergencia alcanzada en iteración " << iter << std::endl;
            break;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now(); // Finalizar medición de tiempo
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "Tiempo de ejecución: " << elapsed.count() << " segundos\n";
}

// Método para imprimir la solución final
void SGD::printSolution() {
    std::cout << "Solución óptima x: ";
    for (double val : x) std::cout << val << " ";
    std::cout << std::endl;
}
