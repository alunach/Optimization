#include "Nesterov.h"

// Constructor: inicializa valores
Nesterov::Nesterov(const std::vector<std::vector<double>>& A, 
                   const std::vector<double>& b, double alpha, 
                   double momentum, int max_iters, double tolerance)
    : A(A), b(b), alpha(alpha), momentum(momentum), max_iters(max_iters), tolerance(tolerance) {
    x.assign(b.size(), 0.0); // Inicializar x en 0
}

// Método de optimización con Nesterov Accelerated Gradient
void Nesterov::optimize() {
    int n = A.size();
    std::vector<double> gradient(n, 0.0);
    std::vector<double> v(n, 0.0); // Vector de momentum
    std::vector<double> x_temp(n, 0.0);
    std::vector<double> Ax(n, 0.0);

    auto start_time = std::chrono::high_resolution_clock::now(); // Iniciar medición de tiempo

    for (int iter = 0; iter < max_iters; iter++) {
        // Calcular x adelantado (lookahead step)
        for (int i = 0; i < n; i++) {
            x_temp[i] = x[i] + momentum * v[i];
        }

        // Calcular Ax usando BLAS: Ax = A * x_temp
        cblas_dgemv(CblasRowMajor, CblasNoTrans, n, n, 1.0, &A[0][0], n, &x_temp[0], 1, 0.0, &Ax[0], 1);

        // Calcular el gradiente en x_temp: grad = A * x_temp - b
        for (int i = 0; i < n; i++) {
            gradient[i] = Ax[i] - b[i];
        }

        // Actualizar momentum v_t = mu * v_{t-1} - alpha * grad
        for (int i = 0; i < n; i++) {
            v[i] = momentum * v[i] - alpha * gradient[i];
        }

        // Actualizar x usando el momentum
        for (int i = 0; i < n; i++) {
            x[i] += v[i];
        }

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
void Nesterov::printSolution() {
    std::cout << "Solución óptima x: ";
    for (double val : x) std::cout << val << " ";
    std::cout << std::endl;
}
