#include "plot_utils.h"
#include <fstream>
#include <iostream>

/**
 * Implementación de la función `plot` para gráficos de líneas.
 */
void PlotUtils::plot(const std::vector<double>& x, const std::vector<double>& y, const std::string& title) {
    std::ofstream data_file("line_data.tmp");
    for (size_t i = 0; i < x.size(); ++i) {
        data_file << x[i] << " " << y[i] << "\n";
    }
    data_file.close();

    std::string command = "gnuplot -e \"set title '" + title + "'; plot 'line_data.tmp' with linespoints; pause -1\"";
    system(command.c_str());
}

/**
 * Implementación de la función `plot_bar` para gráficos de barras.
 */
void PlotUtils::plot_bar(const std::vector<std::string>& labels, const std::vector<double>& values, 
                         const std::string& title, const std::string& xlabel, const std::string& ylabel) {
    std::ofstream data_file("bar_data.tmp");
    for (size_t i = 0; i < labels.size(); ++i) {
        data_file << labels[i] << " " << values[i] << "\n";
    }
    data_file.close();

    std::ofstream script_file("bar_script.gp");
    script_file << "set terminal png size 800,600\n";
    script_file << "set output 'bar_chart.png'\n";
    script_file << "set title \"" << title << "\"\n";
    script_file << "set xlabel \"" << xlabel << "\"\n";
    script_file << "set ylabel \"" << ylabel << "\"\n";
    script_file << "set style data histogram\n";
    script_file << "set style histogram cluster gap 1\n";
    script_file << "set style fill solid border -1\n";
    script_file << "set boxwidth 0.9\n";
    script_file << "set xtics rotate by -45\n";
    script_file << "plot 'bar_data.tmp' using 2:xtic(1) with histogram title columnheader\n";
    script_file.close();

    std::string command = "gnuplot bar_script.gp";
    system(command.c_str());
    std::cout << "Gráfico generado: bar_chart.png" << std::endl;
}
