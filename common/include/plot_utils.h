#ifndef PLOT_UTILS_H
#define PLOT_UTILS_H

#include <vector>
#include <string>

/**
 * Clase para manejar gráficos con Gnuplot.
 */
class PlotUtils {
public:
    /**
     * Genera un gráfico de líneas.
     * @param x Valores del eje X.
     * @param y Valores del eje Y.
     * @param title Título del gráfico.
     */
    static void plot(const std::vector<double>& x, const std::vector<double>& y, const std::string& title);

    /**
     * Genera un gráfico de barras.
     * @param labels Etiquetas de las barras en el eje X.
     * @param values Valores asociados a cada barra en el eje Y.
     * @param title Título del gráfico.
     * @param xlabel Etiqueta para el eje X.
     * @param ylabel Etiqueta para el eje Y.
     */
    static void plot_bar(const std::vector<std::string>& labels, const std::vector<double>& values, 
                         const std::string& title, const std::string& xlabel, const std::string& ylabel);
};

#endif // PLOT_UTILS_H
