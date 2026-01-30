#pragma once
#include <vector>

class ObjectiveFunction {
public:
    virtual double value(const std::vector<double>& x) const = 0;
    virtual void gradient(const std::vector<double>& x,
                          std::vector<double>& grad) const = 0;
    virtual int dimension() const = 0;
    virtual ~ObjectiveFunction() = default;
};
