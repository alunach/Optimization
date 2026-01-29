#include "SVDImage.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <lapacke.h>
#include <cblas.h>

// Constructor: Lee la imagen en escala de grises (PGM P2)
SVDImage::SVDImage(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Error al abrir el archivo: " << filename << std::endl;
        exit(1);
    }

    std::string format;
    file >> format;
    if (format != "P2") {
        std::cerr << "Formato no soportado. Usa una imagen PGM (P2)." << std::endl;
        exit(1);
    }

    file >> width >> height;
    int max_val;
    file >> max_val;

    image_matrix.resize(height, std::vector<double>(width));

    for (int i = 0; i < height; ++i)
        for (int j = 0; j < width; ++j)
            file >> image_matrix[i][j];

    file.close();
}

// Aplica SVD con LAPACK y reduce la imagen usando 'num_singular_values'
void SVDImage::applySVD(int num_singular_values) {
    int m = height;
    int n = width;
    int lda = n;
    int ldu = m;
    int ldvt = n;

    std::vector<double> A(m * n);
    std::vector<double> S(std::min(m, n));
    std::vector<double> U(m * m);
    std::vector<double> VT(n * n);
    std::vector<double> superb(std::min(m, n) - 1);

    // Copiar imagen a matriz A (LAPACK usa formato plano)
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            A[i * n + j] = image_matrix[i][j];

    // Realizar SVD: A = U * S * VT
    int info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', m, n, A.data(), lda, S.data(), 
                              U.data(), ldu, VT.data(), ldvt, superb.data());

    if (info > 0) {
        std::cerr << "Error: No se pudo converger en SVD." << std::endl;
        exit(1);
    }

    // Reconstrucción de la imagen con 'num_singular_values'
    std::vector<double> compressed_A(m * n, 0.0);

    for (int k = 0; k < num_singular_values; ++k)
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                compressed_A[i * n + j] += S[k] * U[i * m + k] * VT[k * n + j];

    // Copiar datos reconstruidos a image_matrix
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            image_matrix[i][j] = std::min(std::max(compressed_A[i * n + j], 0.0), 255.0);
}

// Guarda la imagen comprimida en formato PGM
void SVDImage::saveImage(const std::string& filename) {
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Error al guardar el archivo: " << filename << std::endl;
        exit(1);
    }

    file << "P2\n" << width << " " << height << "\n255\n";
    for (const auto& row : image_matrix) {
        for (double pixel : row)
            file << static_cast<int>(pixel) << " ";
        file << "\n";
    }
    file.close();
}
