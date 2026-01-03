#include <iostream>
#include <eigen3/Eigen/Dense>
#include "../include/core/linear_regression.hpp"

using Eigen::VectorXd;
using Eigen::MatrixXd;

int main(){
	using std::cout; using std::endl;

	LinearRegression lr;

	// Test 1: set_parameters and get_parameters
	VectorXd w(2);
	w << 1.0, 2.0;
	if(lr.set_parameters(w) != 0){
		cout << "set_parameters failed" << endl; return 2;
	}
	VectorXd gw = lr.get_parameters();
	if(gw.size() != 2 || std::abs(gw[0]-1.0) > 1e-12){
		cout << "get_parameters mismatch" << endl; return 3;
	}

	// Test 2: forward
	MatrixXd X(3,2);
	X << 1, 0,
		 0, 1,
		 1, 1;
	auto [yhat_pair, code] = lr.forward(X);
	if(code != 0){ cout << "forward returned error code " << code << endl; return 4; }
	VectorXd yhat = yhat_pair;
	VectorXd expected(3);
	expected << 1.0, 2.0, 3.0;
	if((yhat - expected).norm() > 1e-12){ cout << "forward result incorrect" << endl; return 5; }

	// Test 3: loss
	VectorXd y(3);
	y << 1.0, 2.0, 3.0;
	auto [loss_val, loss_code] = lr.loss(y);
	if(loss_code != 0){ cout << "loss returned code " << loss_code << endl; return 6; }
	if(std::abs(loss_val - 0.0) > 1e-12){ cout << "loss should be zero but is " << loss_val << endl; return 7; }

	// Test 4: gradients (with zero loss gradients should be zero)
	auto [grad, gcode] = lr.gradients(y);
	if(gcode != 0){ cout << "gradients returned code " << gcode << endl; return 8; }
	if(grad.norm() > 1e-12){ cout << "gradients should be zero but are " << grad.transpose() << endl; return 9; }

	cout << "All tests passed" << endl;
	return 0;
}
