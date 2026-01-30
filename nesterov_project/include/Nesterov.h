#pragma once
#include <string>
#include <vector>

class Nesterov {
public:
    struct Result {
        std::vector<double> x;
        int iters = 0;
        double final_f = 0.0;
        double final_grad_norm = 0.0;
        double total_time_ms = 0.0;
    };

    // A: row-major contigua (n*n), b: (n)
    Nesterov(int n,
             std::vector<double> A_row_major,
             std::vector<double> b,
             double alpha,
             double momentum,
             int max_iters,
             double tol_grad);

    Result optimize_to_csv(const std::string& csv_path,
                           const std::vector<double>& x0 = {});

private:
    int n_;
    std::vector<double> A_; // row-major, size n*n
    std::vector<double> b_; // size n

    double alpha_;
    double mu_;
    int max_iters_;
    double tol_grad_;

    double f_value(const std::vector<double>& x,
                   const std::vector<double>& Ax) const;
};
