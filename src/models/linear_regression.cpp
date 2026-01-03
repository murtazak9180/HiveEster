#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <utility>
#include "../../include/core/linear_regression.hpp"


using Eigen::VectorXd; 
using Eigen::MatrixXd; 

/**
 * @param X input feature matrix size (n * m+1)
 * Computes y_hat = X * W - bias included as 1 vector in X. 
 * @return predictions y_hat as VectorXd based on current weights
 */
std::pair<VectorXd, int> LinearRegression::forward(const MatrixXd& X){
    if(this->weights.size() == 0 || X.size() == 0){
        //weights are not set, must return error. 
        return std::make_pair(VectorXd(), -1);
    }
    if(this->weights.size() != X.cols()){
        return std::make_pair(VectorXd(), -2); 
    }
    this->X_cached = X; 
    this->y_hat_cached = X * this->weights; 
    return std::make_pair(this->y_hat_cached, 0); 
}

/**
 * @param the ground truth vector Y size n *1 
 * computes the MSE loss
 * @return double loss 
 */

std::pair<double, int>  LinearRegression::loss(const VectorXd& y) const{
    if(y.size() == 0 || this->y_hat_cached.size() == 0){
        return std::make_pair(double(), -1);
    }

    if(y.size() != this->y_hat_cached.size()){
        return std::make_pair(double(), -2);
    }
    //Computes the MSE loss : 1/2N * (Y - Y_hat)^2
    double loss = (((y - this->y_hat_cached).array().square()).sum()) / (this->y_hat_cached.size() *2); 
    return std::make_pair(loss, 0); 

}

std::pair<VectorXd, int>  LinearRegression:: gradients(const VectorXd& y) const{
  if(y.size() == 0 || this->y_hat_cached.size() == 0){
    return std::make_pair(VectorXd(),-1);
  }
  if(this->weights.size() == 0){
    return std::make_pair(VectorXd(),-2);
  }

  if(this->y_hat_cached.size() != y.size()){
    return std::make_pair(VectorXd(), -3);
  }

  VectorXd grad = (this->X_cached.transpose()* (this->y_hat_cached - y))/this->y_hat_cached.size();

  return std::make_pair(grad, 0);
}

VectorXd LinearRegression::get_parameters() const{
    return this->weights; 
}

int LinearRegression::set_parameters(const VectorXd& params){
    if(params.size() == 0){
        return -1; 
    }

    this->weights = params; 
    return 0;
}
