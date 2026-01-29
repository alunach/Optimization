#include "read_csv.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <random>
#include <algorithm>
#include <numeric>

std::vector<std::vector<double>> CSVReader::read(const std::string& file_path, std::vector<int>& labels) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("No se pudo abrir el archivo CSV.");
    }

    std::string line;

    // Ignorar la primera línea (el encabezado)
    std::getline(file, line);

    std::vector<std::vector<double>> data;

    while (std::getline(file, line)) {
        if (line.empty()) continue; // Ignorar líneas vacías

        std::stringstream ss(line);
        std::string value;
        std::vector<double> row;

        while (std::getline(ss, value, ',')) {
            try {
                row.push_back(std::stod(value));
            } catch (const std::invalid_argument&) {
                std::cerr << "Error: Valor no numérico encontrado en el CSV: " << value << std::endl;
                throw;
            }
        }

        labels.push_back(static_cast<int>(row[0])); // La primera columna es la etiqueta
        row.erase(row.begin()); // Elimina la etiqueta de los datos
        data.push_back(row);
    }

    return data;
}

std::pair<std::pair<std::vector<std::vector<double>>, std::vector<int>>,
          std::pair<std::vector<std::vector<double>>, std::vector<int>>>
CSVReader::split(const std::vector<std::vector<double>>& data, const std::vector<int>& labels, double train_ratio) {
    if (data.size() != labels.size()) {
        throw std::runtime_error("El tamaño de los datos y las etiquetas no coincide.");
    }

    // Crear índices aleatorios
    std::vector<size_t> indices(data.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    size_t train_size = static_cast<size_t>(data.size() * train_ratio);

    // Dividir datos y etiquetas
    std::vector<std::vector<double>> train_data, test_data;
    std::vector<int> train_labels, test_labels;

    for (size_t i = 0; i < data.size(); ++i) {
        if (i < train_size) {
            train_data.push_back(data[indices[i]]);
            train_labels.push_back(labels[indices[i]]);
        } else {
            test_data.push_back(data[indices[i]]);
            test_labels.push_back(labels[indices[i]]);
        }
    }

    return {{train_data, train_labels}, {test_data, test_labels}};
}

