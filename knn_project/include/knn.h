#ifndef KNN_H
#define KNN_H

#include <vector>
#include <functional>
#include <string>

/**
 * Clase para implementar el algoritmo k-Nearest Neighbors con soporte para diferentes métricas de distancia.
 */
class KNN {
public:
    /**
     * Constructor de la clase KNN.
     * @param k Número de vecinos a considerar.
     */
    explicit KNN(int k);

    /**
     * Clasifica un conjunto de datos de prueba utilizando la métrica de distancia especificada.
     * @param X_train Datos de entrenamiento.
     * @param y_train Etiquetas de entrenamiento.
     * @param X_test Punto de prueba.
     * @param distance_fn Función para calcular la distancia.
     * @return Etiqueta predicha.
     */
    int classify(const std::vector<std::vector<double>>& X_train,
                 const std::vector<int>& y_train,
                 const std::vector<double>& X_test,
                 std::function<double(const std::vector<double>&, const std::vector<double>&)> distance_fn) const;

    /**
     * Métodos estáticos para las métricas de distancia.
     */
    static double euclidean_distance(const std::vector<double>& a, const std::vector<double>& b);
    static double manhattan_distance(const std::vector<double>& a, const std::vector<double>& b);
    static double cosine_distance(const std::vector<double>& a, const std::vector<double>& b);
    static double chebyshev_distance(const std::vector<double>& a, const std::vector<double>& b);
    static double minkowski_distance(const std::vector<double>& a, const std::vector<double>& b, int p);

private:
    int k;
};

#endif // KNN_H
