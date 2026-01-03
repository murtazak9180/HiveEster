#pragma once
#include <eigen3/Eigen/Dense>
#include <utility>

using Eigen::VectorXd;
using Eigen::MatrixXd;

class Model {
public:
    // Forward pass 
    virtual std::pair<VectorXd, int> forward(const MatrixXd& X) = 0;
    // Loss computed from last forward
    virtual std::pair<double, int> loss(const VectorXd& y) const = 0;
    // Gradients from last forward + given labels
    virtual std::pair<VectorXd, int> gradients(const VectorXd& y) const = 0;

    virtual VectorXd get_parameters() const = 0;
    virtual int set_parameters(const VectorXd& params) = 0;

    virtual ~Model() {}
};
