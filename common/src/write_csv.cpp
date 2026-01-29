#include "write_csv.h"
#include <fstream>

void CSVWriter::write(const std::string& filename, const std::vector<std::vector<double>>& data, const std::vector<std::string>& headers) {
    std::ofstream file(filename);

    // Escribir encabezados
    for (size_t i = 0; i < headers.size(); ++i) {
        file << headers[i];
        if (i < headers.size() - 1) file << ",";
    }
    file << "\n";

    // Escribir datos
    for (const auto& row : data) {
        for (size_t i = 0; i < row.size(); ++i) {
            file << row[i];
            if (i < row.size() - 1) file << ",";
        }
        file << "\n";
    }
}
