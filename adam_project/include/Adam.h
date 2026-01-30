#pragma once
#include <string>
#include <vector>

class Adam {
public:
    struct Result {
        std::vector<double> x;
        int iters = 0;
        double final_f = 0.0;
        double final_grad_norm = 0.0;
        double total_time_ms = 0.0;
    };

    // A: row-major contigua (n*n), b: (n)
    Adam(int n,
         std::vector<double> A_row_major,
         std::vector<double> b,
         double alpha,
         double beta1,
         double beta2,
         double eps,
         int max_iters,
         double tol_grad);

    Result optimize_to_csv(const std::string& csv_path,
                           const std::vector<double>& x0 = {});

private:
    int n_;
    std::vector<double> A_; // row-major, size n*n
    std::vector<double> b_; // size n

    double alpha_;
    double beta1_;
    double beta2_;
    double eps_;
    int max_iters_;
    double tol_grad_;

    double f_value(const std::vector<double>& x,
                   const std::vector<double>& Ax) const;
};
