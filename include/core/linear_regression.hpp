#pragma once 
#include <eigen3/Eigen/Dense>
#include <utility>

#include "model.h"
#include "../utils/utils.hpp"



using Eigen::VectorXd; 
using Eigen::MatrixXd; 


class LinearRegression: public Model{
    private:
    VectorXd weights;   // ((m+1)*1) weights . bias included in the feature vector as all 1's 

    VectorXd y_hat_cached; // cache the predictions and the current feature matrix.  
    MatrixXd X_cached;  // (n * m+1) feature matrix

    public:
    //constructor and destructor aren't really needed right now. 
    std::pair<VectorXd, int> forward(const MatrixXd& X) override; 
    std::pair<double, int>  loss(const VectorXd& y) const override; 
    std::pair<VectorXd, int> gradients(const VectorXd& y) const override; 
    VectorXd get_parameters() const override; 
    int set_parameters(const VectorXd& params) override; 
    
};
