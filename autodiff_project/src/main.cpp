#include "autodiff.h"
#include <iostream>

int main() {
    std::vector<std::vector<double>> W = {{1.0, 2.0}, {3.0, 4.0}};
    std::vector<double> x = {1.0, -1.0};

    AutoDiff autodiff(W, x);

    autodiff.forward_and_backward();
    autodiff.printResults();

    return 0;
}
