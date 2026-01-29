#include "knn.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>

// Constructor
KNN::KNN(int k) : k(k) {}

// Clasificación
int KNN::classify(const std::vector<std::vector<double>>& X_train,
                  const std::vector<int>& y_train,
                  const std::vector<double>& X_test,
                  std::function<double(const std::vector<double>&, const std::vector<double>&)> distance_fn) const {
    if (!distance_fn) {
        throw std::invalid_argument("No se proporcionó una función de distancia.");
    }

    // Calcular distancias
    std::vector<std::pair<double, int>> distance_label_pairs;
    for (size_t i = 0; i < X_train.size(); ++i) {
        double distance = distance_fn(X_train[i], X_test);
        distance_label_pairs.emplace_back(distance, y_train[i]);
    }

    // Ordenar por distancia
    std::sort(distance_label_pairs.begin(), distance_label_pairs.end());

    // Contar las etiquetas de los k vecinos más cercanos
    std::vector<int> label_count(*std::max_element(y_train.begin(), y_train.end()) + 1, 0);
    for (int i = 0; i < k; ++i) {
        label_count[distance_label_pairs[i].second]++;
    }

    // Devolver la etiqueta con mayor frecuencia
    return std::distance(label_count.begin(), std::max_element(label_count.begin(), label_count.end()));
}

// Distancia Euclidiana
double KNN::euclidean_distance(const std::vector<double>& a, const std::vector<double>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return std::sqrt(sum);
}

// Distancia Manhattan
double KNN::manhattan_distance(const std::vector<double>& a, const std::vector<double>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += std::abs(a[i] - b[i]);
    }
    return sum;
}

// Distancia Coseno
double KNN::cosine_distance(const std::vector<double>& a, const std::vector<double>& b) {
    double dot_product = std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
    double magnitude_a = std::sqrt(std::inner_product(a.begin(), a.end(), a.begin(), 0.0));
    double magnitude_b = std::sqrt(std::inner_product(b.begin(), b.end(), b.begin(), 0.0));
    return 1.0 - (dot_product / (magnitude_a * magnitude_b));
}

// Distancia Chebyshev
double KNN::chebyshev_distance(const std::vector<double>& a, const std::vector<double>& b) {
    double max_diff = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
    }
    return max_diff;
}

// Distancia Minkowski
double KNN::minkowski_distance(const std::vector<double>& a, const std::vector<double>& b, int p) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += std::pow(std::abs(a[i] - b[i]), p);
    }
    return std::pow(sum, 1.0 / p);
}
