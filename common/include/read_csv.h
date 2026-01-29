#ifndef READ_CSV_H
#define READ_CSV_H

#include <vector>
#include <string>

/**
 * Clase para leer datos de un archivo CSV.
 */
class CSVReader {
public:
    /**
     * Lee un archivo CSV y devuelve los datos.
     * @param file_path Ruta al archivo CSV.
     * @param labels Vector para almacenar las etiquetas.
     * @return Matriz de datos numéricos.
     */
    static std::vector<std::vector<double>> read(const std::string& file_path, std::vector<int>& labels);
     /**
     * Divide los datos en conjuntos de entrenamiento y prueba.
     * @param data Matriz de datos numéricos.
     * @param labels Vector de etiquetas correspondientes.
     * @param train_ratio Proporción de datos para entrenamiento (por ejemplo, 0.8).
     * @return Un par de vectores: {datos_entrenamiento, datos_prueba}.
     */
    static std::pair<std::pair<std::vector<std::vector<double>>, std::vector<int>>,
                     std::pair<std::vector<std::vector<double>>, std::vector<int>>>
    split(const std::vector<std::vector<double>>& data, const std::vector<int>& labels, double train_ratio = 0.8);
};

#endif // READ_CSV_H
