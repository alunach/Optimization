#include "Adam.h"

// Constructor: inicializa valores
Adam::Adam(const std::vector<std::vector<double>>& A, 
           const std::vector<double>& b, double alpha, 
           double beta1, double beta2, double epsilon, 
           int max_iters, double tolerance)
    : A(A), b(b), alpha(alpha), beta1(beta1), beta2(beta2), 
      epsilon(epsilon), max_iters(max_iters), tolerance(tolerance) {
    x.assign(b.size(), 0.0); // Inicializar x en 0
}

// Método de optimización con Adam
void Adam::optimize() {
    int n = A.size();
    std::vector<double> gradient(n, 0.0);
    std::vector<double> m(n, 0.0);
    std::vector<double> v(n, 0.0);
    std::vector<double> Ax(n, 0.0);

    auto start_time = std::chrono::high_resolution_clock::now(); // Iniciar medición de tiempo

    for (int iter = 1; iter <= max_iters; iter++) {
        // Calcular Ax usando BLAS: Ax = A * x
        cblas_dgemv(CblasRowMajor, CblasNoTrans, n, n, 1.0, &A[0][0], n, &x[0], 1, 0.0, &Ax[0], 1);

        // Calcular el gradiente: grad = A * x - b
        for (int i = 0; i < n; i++)
            gradient[i] = Ax[i] - b[i];

        // Calcular momentos Adam
        for (int i = 0; i < n; i++) {
            m[i] = beta1 * m[i] + (1 - beta1) * gradient[i];
            v[i] = beta2 * v[i] + (1 - beta2) * (gradient[i] * gradient[i]);
        }

        // Corregir sesgo
        std::vector<double> m_hat(n, 0.0);
        std::vector<double> v_hat(n, 0.0);
        for (int i = 0; i < n; i++) {
            m_hat[i] = m[i] / (1 - std::pow(beta1, iter));
            v_hat[i] = v[i] / (1 - std::pow(beta2, iter));
        }

        // Actualizar x: x = x - alpha * m_hat / (sqrt(v_hat) + epsilon)
        for (int i = 0; i < n; i++) {
            x[i] -= alpha * m_hat[i] / (std::sqrt(v_hat[i]) + epsilon);
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
void Adam::printSolution() {
    std::cout << "Solución óptima x: ";
    for (double val : x) std::cout << val << " ";
    std::cout << std::endl;
}
