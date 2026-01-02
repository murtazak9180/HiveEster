#pragma once
#include <eigen3/Eigen/Dense>

using Eigen::VectorXd;
using Eigen::MatrixXd;

class Model {
public:
    // Forward pass: compute predictions
    virtual VectorXd forward(const MatrixXd& X) const = 0;

    // Compute loss for given X and labels y
    virtual double loss(const MatrixXd& X, const VectorXd& y) const = 0;

    // Compute gradients w.r.t model parameters
    virtual VectorXd gradients(const MatrixXd& X, const VectorXd& y) const = 0;

    // Get model parameters as a single vector
    virtual VectorXd get_parameters() const = 0;

    // Set model parameters from a vector
    virtual void set_parameters(const VectorXd& params) = 0;

    virtual ~Model() {}
};
