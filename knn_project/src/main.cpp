#include <iostream>
#include "knn.h"
#include "read_csv.h"
#include "plot_utils.h"

/**
 * Calcula la precisión del modelo kNN.
 * @param knn Instancia del modelo kNN.
 * @param X_train Datos de entrenamiento.
 * @param y_train Etiquetas de entrenamiento.
 * @param X_test Datos de prueba.
 * @param y_test Etiquetas de prueba.
 * @param distance_fn Función de distancia.
 * @return Precisión del modelo.
 */
double calculate_accuracy(KNN& knn, 
                          const std::vector<std::vector<double>>& X_train, 
                          const std::vector<int>& y_train, 
                          const std::vector<std::vector<double>>& X_test, 
                          const std::vector<int>& y_test, 
                          std::function<double(const std::vector<double>&, const std::vector<double>&)> distance_fn) {
    int correct_predictions = 0;

    for (size_t i = 0; i < X_test.size(); ++i) {
        int predicted_label = knn.classify(X_train, y_train, X_test[i], distance_fn);
        if (predicted_label == y_test[i]) {
            ++correct_predictions;
        }
    }

    return static_cast<double>(correct_predictions) / X_test.size();
}

int main() {
    std::cout << "Evaluación del modelo kNN con múltiples distancias." << std::endl;

    // Leer datos del archivo CSV
    std::vector<int> labels;
    auto data = CSVReader::read("../data/wine.csv", labels);

    // Dividir datos en entrenamiento y prueba
    auto [train, test] = CSVReader::split(data, labels, 0.8);
    auto train_data = train.first;
    auto train_labels = train.second;
    auto test_data = test.first;
    auto test_labels = test.second;

    // Instancia del modelo kNN
    KNN knn(3);

    // Evaluar con diferentes métricas de distancia
    std::cout << "Calculando precisión..." << std::endl;

    // Precisión con distancia Euclidiana
    double euclidean_accuracy = calculate_accuracy(knn, train_data, train_labels, test_data, test_labels, KNN::euclidean_distance);
    std::cout << "Precisión (Euclidiana): " << euclidean_accuracy * 100.0 << "%" << std::endl;

    // Precisión con distancia Manhattan
    double manhattan_accuracy = calculate_accuracy(knn, train_data, train_labels, test_data, test_labels, KNN::manhattan_distance);
    std::cout << "Precisión (Manhattan): " << manhattan_accuracy * 100.0 << "%" << std::endl;

    // Precisión con distancia Coseno
    double cosine_accuracy = calculate_accuracy(knn, train_data, train_labels, test_data, test_labels, KNN::cosine_distance);
    std::cout << "Precisión (Coseno): " << cosine_accuracy * 100.0 << "%" << std::endl;

    // Graficar resultados
    std::vector<std::string> metrics = {"Euclidiana", "Manhattan", "Coseno"};
    std::vector<double> accuracies = {euclidean_accuracy * 100.0, manhattan_accuracy * 100.0, cosine_accuracy * 100.0};

    std::cout << "Generando gráfico de precisión..." << std::endl;
    PlotUtils::plot_bar(metrics, accuracies, "Precisión de kNN con diferentes distancias", "Métrica", "Precisión (%)");

    return 0;
}
