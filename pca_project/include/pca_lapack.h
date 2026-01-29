#ifndef PCA_LAPACK_H
#define PCA_LAPACK_H

#include <vector>

class PCA {
public:
    PCA(int num_components);
    std::vector<std::vector<double>> fit_transform(const std::vector<std::vector<double>>& data);

private:
    int num_components;
};

#endif // PCA_LAPACK_H
