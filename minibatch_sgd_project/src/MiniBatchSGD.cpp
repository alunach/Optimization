#include "MiniBatchSGD.h"

// Constructor: inicializa valores
MiniBatchSGD::MiniBatchSGD(const std::vector<std::vector<double>>& A, 
                           const std::vector<double>& b, double alpha, 
                           int max_iters, double tolerance, int batch_size)
    : A(A), b(b), alpha(alpha), max_iters(max_iters), tolerance(tolerance), batch_size(batch_size) {
    x.assign(b.size(), 0.0); // Inicializar x en 0
}

// Método de optimización con Mini-Batch SGD
void MiniBatchSGD::optimize() {
    int n = A.size();
    std::vector<double> Ax(n, 0.0);
    std::vector<double> gradient(n, 0.0);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, n - 1);

    auto start_time = std::chrono::high_resolution_clock::now(); // Iniciar medición de tiempo

    for (int iter = 0; iter < max_iters; iter++) {
        // Seleccionar un mini-batch aleatorio
        std::vector<int> indices;
        for (int i = 0; i < batch_size; i++) {
            indices.push_back(dist(gen));
        }

        // Resetear gradiente acumulado
        std::fill(gradient.begin(), gradient.end(), 0.0);

        // Procesar cada muestra en el mini-batch
        for (int i : indices) {
            // Calcular Ax para la muestra seleccionada
            Ax[i] = cblas_ddot(n, &A[i][0], 1, &x[0], 1);

            // Calcular el gradiente para la muestra i
            double grad_i = Ax[i] - b[i];

            // Acumular el gradiente
            for (int j = 0; j < n; j++) {
                gradient[j] += grad_i * A[i][j] / batch_size;  // Promediar por el tamaño del batch
            }
        }

        // Actualizar x con el gradiente del mini-batch
        cblas_daxpy(n, -alpha, &gradient[0], 1, &x[0], 1);

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
void MiniBatchSGD::printSolution() {
    std::cout << "Solución óptima x: ";
    for (double val : x) std::cout << val << " ";
    std::cout << std::endl;
}
