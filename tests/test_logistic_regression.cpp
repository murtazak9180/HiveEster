#include <iostream>
#include <eigen3/Eigen/Dense>
#include "../include/core/logistic_regression.hpp"

using Eigen::VectorXd;
using Eigen::MatrixXd;

int main(){
    using std::cout; using std::endl;

    LogisticRegression lr;

    // Test 1: set_parameters and get_parameters
    VectorXd w(2);
    w << 0.0, 0.0; // zero weights -> sigmoid(0) == 0.5
    if(lr.set_parameters(w) != 0){
        cout << "set_parameters failed" << endl; return 2;
    }
    VectorXd gw = lr.get_parameters();
    if(gw.size() != 2 || std::abs(gw[0]-0.0) > 1e-12){
        cout << "get_parameters mismatch" << endl; return 3;
    }

    // Test 2: forward -> all probabilities should be 0.5
    MatrixXd X(3,2);
    X << 1, 0,
         0, 1,
         1, 1;
    auto [yhat_pair, code] = lr.forward(X);
    if(code != 0){ cout << "forward returned error code " << code << endl; return 4; }
    VectorXd yhat = yhat_pair;
    VectorXd expected(3);
    expected << 0.5, 0.5, 0.5;
    if((yhat - expected).norm() > 1e-12){ cout << "forward result incorrect: " << yhat.transpose() << endl; return 5; }

    // Test 3: loss (binary cross-entropy) for labels [1,0,1]
    VectorXd y(3);
    y << 1.0, 0.0, 1.0;
    auto [loss_val, loss_code] = lr.loss(y);
    if(loss_code != 0){ cout << "loss returned code " << loss_code << endl; return 6; }
    double expected_loss = -std::log(0.5); // each sample loss = ln2, mean = ln2
    if(std::abs(loss_val - expected_loss) > 1e-9){ cout << "loss incorrect: " << loss_val << " expected " << expected_loss << endl; return 7; }

    // Test 4: gradients
    auto [grad, gcode] = lr.gradients(y);
    if(gcode != 0){ cout << "gradients returned code " << gcode << endl; return 8; }
    // For w=0 and X as above, residual = y_hat - y = [-0.5, 0.5, -0.5]
    // grad = X^T * residual / 3 = [-1/3, 0]
    VectorXd expected_grad(2);
    expected_grad << -1.0/3.0, 0.0;
    if((grad - expected_grad).norm() > 1e-9){ cout << "gradients incorrect: " << grad.transpose() << " expected " << expected_grad.transpose() << endl; return 9; }

    cout << "All logistic tests passed" << endl;
    return 0;
}
