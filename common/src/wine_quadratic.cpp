// common/src/wine_quadratic.cpp
#include "wine_quadratic.h"

#include <cblas.h>
#include <lapacke.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

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
                                               [](unsigned char ch) { return !std::isspace(ch); }));
            tok.erase(std::find_if(tok.rbegin(), tok.rend(),
                                   [](unsigned char ch) { return !std::isspace(ch); })
                          .base(),
                      tok.end());
            tokens.push_back(tok);
        }
        if (tokens.size() < 2) continue;

        if (!header_checked) {
            header_checked = true;
            bool all_numeric = true;
            for (auto& t : tokens) {
                if (!is_number(t)) { all_numeric = false; break; }
            }
            if (!all_numeric) continue; // salta header
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

static void zscore_normalize(int m, int d, std::vector<double>& X) {
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
        for (int i = 0; i < m; ++i)
            X[(size_t)i * d + j] = (X[(size_t)i * d + j] - mean[j]) / stdv;
    }
}

QuadraticAB build_quadratic_from_wine_csv(
    const std::string& path,
    double lambda,
    bool normalize,
    int y_mode,
    int positive_class
) {
    auto rows = read_csv_numeric(path);

    const int m = (int)rows.size();
    const int p = (int)rows[0].size();
    if (p < 2) throw std::runtime_error("wine.csv debe tener >=2 columnas (label + features)");

    // ✅ wine.csv: col0 = label, col1.. = features
    const int d = p - 1;

    std::vector<double> X((size_t)m * d, 0.0);
    std::vector<double> y(m, 0.0);

    for (int i = 0; i < m; ++i) {
        if ((int)rows[i].size() != p)
            throw std::runtime_error("Filas con distinto número de columnas en CSV");

        double label = rows[i][0]; // ✅ PRIMERA columna
        if (y_mode == 0) {
            y[i] = label; // (regresión sobre la clase como double)
        } else {
            y[i] = ((int)std::lround(label) == positive_class) ? 1.0 : 0.0;
        }

        for (int j = 0; j < d; ++j) {
            X[(size_t)i * d + j] = rows[i][j + 1]; // ✅ features desde col1
        }
    }

    if (normalize) zscore_normalize(m, d, X);

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

static void symmetrize_inplace(int n, std::vector<double>& A) {
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            double v = 0.5 * (A[(size_t)i * n + j] + A[(size_t)j * n + i]);
            A[(size_t)i * n + j] = v;
            A[(size_t)j * n + i] = v;
        }
    }
}

QuadraticAB build_quadratic_case_from_wine_csv(
    const std::string& path,
    QuadraticCase which,
    double lambda,
    bool normalize,
    int y_mode,
    int positive_class,
    double illcond_ratio,
    double nonconvex_delta
) {
    // Base SPD
    QuadraticAB base = build_quadratic_from_wine_csv(path, lambda, normalize, y_mode, positive_class);
    const int n = base.n;

    // A0 = X^T X (PSD): restando lambda I si lambda>0
    auto A0 = base.A;
    if (lambda > 0.0) {
        for (int i = 0; i < n; ++i) A0[(size_t)i * n + i] -= lambda;
    }
    symmetrize_inplace(n, A0);

    QuadraticAB out = base;

    switch (which) {
        case QuadraticCase::ConvexPSD: {
            out.A = std::move(A0);
            break;
        }

        case QuadraticCase::StronglyConvexSPD: {
            // ya es SPD por construcción
            break;
        }

        case QuadraticCase::IllConditionedSPD: {
            if (illcond_ratio < 1.0) illcond_ratio = 1.0;

            std::vector<double> Awork = out.A;
            std::vector<double> w(n, 0.0);

            int info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', n, Awork.data(), n, w.data());
            if (info != 0) throw std::runtime_error("LAPACKE_dsyev fallo (IllConditionedSPD), info=" + std::to_string(info));

            const double lmax = w.back();
            const double target_min = lmax / illcond_ratio;

            std::vector<double> s(n, 0.0);
            for (int i = 0; i < n; ++i) {
                double t = (n == 1) ? 0.0 : (double)i / (double)(n - 1);
                s[i] = std::exp(std::log(target_min) * (1.0 - t) + std::log(lmax) * t);
            }

            std::vector<double> B = Awork; // Q
            for (int i = 0; i < n; ++i) {
                double scale = std::sqrt(s[i]);
                for (int j = 0; j < n; ++j) B[(size_t)j * n + i] *= scale;
            }

            std::vector<double> Anew((size_t)n * n, 0.0);
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        n, n, n,
                        1.0, B.data(), n,
                        B.data(), n,
                        0.0, Anew.data(), n);

            symmetrize_inplace(n, Anew);
            out.A = std::move(Anew);
            break;
        }

        case QuadraticCase::NonConvexIndefinite: {
            std::vector<double> Awork = out.A;
            std::vector<double> w(n, 0.0);

            int info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', n, Awork.data(), n, w.data());
            if (info != 0) throw std::runtime_error("LAPACKE_dsyev fallo (NonConvexIndefinite), info=" + std::to_string(info));

            const double lambda_min = w.front();
            const double delta = std::max(nonconvex_delta, 0.0) + lambda_min + 1e-3;

            std::vector<double> v(n, 0.0);
            for (int i = 0; i < n; ++i) v[i] = Awork[(size_t)i * n + 0];

            std::vector<double> Anew = out.A;
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    Anew[(size_t)i * n + j] -= delta * v[i] * v[j];
                }
            }
            symmetrize_inplace(n, Anew);
            out.A = std::move(Anew);
            break;
        }
        default:
            break;
    }

    return out;
}
