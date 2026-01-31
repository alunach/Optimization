#include "wine_quadratic.h"
#include <cblas.h>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cmath>
#include <algorithm>

static bool is_number(const std::string& s) {
    if (s.empty()) return false;
    char* end = nullptr;
    std::strtod(s.c_str(), &end);
    return end != s.c_str() && *end == '\0';
}

static std::vector<std::vector<double>> read_csv_numeric(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open()) throw std::runtime_error("No se pudo abrir CSV: " + path);

    std::vector<std::vector<double>> rows;
    std::string line;

    bool header_checked = false;

    while (std::getline(in, line)) {
        if (line.empty()) continue;

        std::vector<std::string> tokens;
        std::stringstream ss(line);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            tok.erase(tok.begin(), std::find_if(tok.begin(), tok.end(),
                [](unsigned char ch){ return !std::isspace(ch); }));
            tok.erase(std::find_if(tok.rbegin(), tok.rend(),
                [](unsigned char ch){ return !std::isspace(ch); }).base(), tok.end());
            tokens.push_back(tok);
        }
        if (tokens.size() < 2) continue;

        if (!header_checked) {
            header_checked = true;
            bool all_numeric = true;
            for (auto& t : tokens) {
                if (!is_number(t)) { all_numeric = false; break; }
            }
            if (!all_numeric) continue; // skip header
        }

        std::vector<double> r;
        r.reserve(tokens.size());
        for (auto& t : tokens) {
            if (!is_number(t)) { r.clear(); break; }
            r.push_back(std::stod(t));
        }
        if (!r.empty()) rows.push_back(std::move(r));
    }

    if (rows.empty()) throw std::runtime_error("CSV sin filas numéricas: " + path);
    return rows;
}

QuadraticAB build_quadratic_from_csv(
    const std::string& path,
    double lambda,
    bool normalize,
    int y_mode,
    int positive_class
) {
    auto rows = read_csv_numeric(path);

    const int m = (int)rows.size();
    const int p = (int)rows[0].size();
    if (p < 2) throw std::runtime_error("CSV debe tener >=2 columnas");

    int d = 0;
    std::vector<double> X;
    std::vector<double> y(m, 0.0);

    if (y_mode == 0) {
        // Regresión: y = última columna, X = resto
        d = p - 1;
        X.assign((size_t)m * d, 0.0);

        for (int i = 0; i < m; ++i) {
            if ((int)rows[i].size() != p) throw std::runtime_error("Filas con distinto número de columnas");
            for (int j = 0; j < d; ++j) X[(size_t)i * d + j] = rows[i][j];
            y[i] = rows[i][d];
        }
    } else {
        // One-vs-rest Wine: label = primera columna, X = columnas 1..p-1
        d = p - 1;
        X.assign((size_t)m * d, 0.0);

        for (int i = 0; i < m; ++i) {
            if ((int)rows[i].size() != p) throw std::runtime_error("Filas con distinto número de columnas");

            double label = rows[i][0];
            y[i] = ((int)std::lround(label) == positive_class) ? 1.0 : 0.0;

            for (int j = 0; j < d; ++j) {
                X[(size_t)i * d + j] = rows[i][j + 1];
            }
        }
    }

    // Normalización z-score
    if (normalize) {
        std::vector<double> mean(d, 0.0), var(d, 0.0);

        for (int j = 0; j < d; ++j) {
            double s = 0.0;
            for (int i = 0; i < m; ++i) s += X[(size_t)i * d + j];
            mean[j] = s / m;
        }
        for (int j = 0; j < d; ++j) {
            double s = 0.0;
            for (int i = 0; i < m; ++i) {
                double z = X[(size_t)i * d + j] - mean[j];
                s += z * z;
            }
            var[j] = s / std::max(1, m - 1);
        }
        for (int j = 0; j < d; ++j) {
            double stdv = std::sqrt(std::max(var[j], 1e-12));
            for (int i = 0; i < m; ++i) {
                X[(size_t)i * d + j] = (X[(size_t)i * d + j] - mean[j]) / stdv;
            }
        }
    }

    // A = X^T X + lambda I
    std::vector<double> A((size_t)d * d, 0.0);
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                d, d, m,
                1.0, X.data(), d,
                X.data(), d,
                0.0, A.data(), d);

    if (lambda > 0.0) {
        for (int i = 0; i < d; ++i) A[(size_t)i * d + i] += lambda;
    }

    // b = X^T y
    std::vector<double> b(d, 0.0);
    cblas_dgemv(CblasRowMajor, CblasTrans,
                m, d,
                1.0, X.data(), d,
                y.data(), 1,
                0.0, b.data(), 1);

    QuadraticAB out;
    out.n = d;
    out.A = std::move(A);
    out.b = std::move(b);
    return out;
}
