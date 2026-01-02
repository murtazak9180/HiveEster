#pragma once 
#include <eigen3/Eigen/Dense>

#include "model.h"
#include "../utils/utils.h"

using Eigen::VectorXd; 
using Eigen::MatrixXd; 




class LinearRegression: public Model{
    private:
    VectorXd weights;   //bias included in the feature vector as all 1's 

    public:
    LinearRegression(int n); //How can we know the number of features i.e weights vector size?
                             // is it set with client? does server set it? 
    
};
