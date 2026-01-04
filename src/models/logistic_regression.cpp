#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <utility>
#include "../../include/core/logistic_regression.hpp"


using Eigen::VectorXd; 
using Eigen::MatrixXd; 



std::pair<VectorXd, int> LogisticRegression::sigmoid(const VectorXd& z){
      if(z.size() == 0){
        return std::make_pair(VectorXd(), -1);
      }
      return std::make_pair(1.0 / (1.0 + (-z.array()).exp()), 0);
}




/**
 * @param X input feature matrix size (n * m+1)
 * Computes y_hat = sigmoid(X * W) - bias included as 1 vector in X. 
 * @return predictions y_hat as VectorXd based on current weights
 */
std::pair<VectorXd, int> LogisticRegression::forward(const MatrixXd& X){
    if(this->weights.size() == 0 || X.size() == 0){
        //weights are not set, must return error. 
        return std::make_pair(VectorXd(), -1);
    }
    if(this->weights.size() != X.cols()){
        return std::make_pair(VectorXd(), -2); 
    }
    this->X_cached = X; 
    auto [y_hat, status] = this->sigmoid(X * this->weights); 
    if(status){
        return std::make_pair(VectorXd(), -3);
    }
    this->y_hat_cached = y_hat; 
    return std::make_pair(this->y_hat_cached, 0); 
}

/**
 * @param the ground truth vector Y size n *1 
 * computes the MSE loss
 * @return double loss 
 */

std::pair<double, int>  LogisticRegression::loss(const VectorXd& y) const{
    if(y.size() == 0 || this->y_hat_cached.size() == 0){
        return std::make_pair(double(), -1);
    }

    if(y.size() != this->y_hat_cached.size()){
        return std::make_pair(double(), -2);
    }
    //

    auto pa = this->y_hat_cached.array();
    auto ya = y.array();
    double loss = - ( ya * pa.log() + (1 - ya) * (1 - pa).log() ).mean();

    return std::make_pair(loss,0);
}

std::pair<VectorXd, int>  LogisticRegression:: gradients(const VectorXd& y) const{
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

VectorXd LogisticRegression::get_parameters() const{
    return this->weights; 
}

int LogisticRegression::set_parameters(const VectorXd& params){
    if(params.size() == 0){
        return -1; 
    }

    this->weights = params; 
    return 0;
}