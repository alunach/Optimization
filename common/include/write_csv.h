#ifndef WRITE_CSV_H
#define WRITE_CSV_H

#include <vector>
#include <string>

/**
 * Clase para escribir datos en un archivo CSV.
 */
class CSVWriter {
public:
    /**
     * Escribe datos en un archivo CSV.
     * @param filename Nombre del archivo de salida.
     * @param data Datos a escribir.
     * @param headers Encabezados de las columnas.
     */
    static void write(const std::string& filename, const std::vector<std::vector<double>>& data, const std::vector<std::string>& headers);
};

#endif // WRITE_CSV_H
