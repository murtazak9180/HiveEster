#include<eigen3/Eigen/Dense>
#include "model.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;

class LogisticRegression: public Model{
    private:
    VectorXd weights; 
    VectorXd y_hat_cached; 
    MatrixXd X_cached;


    std::pair<VectorXd, int> sigmoid(const VectorXd &z);
    VectorXd element_wise_log(VectorXd&z);
    public:
    std::pair<VectorXd, int> forward(const MatrixXd& X) override; 
    std::pair<double, int>  loss(const VectorXd& y) const override; 
    std::pair<VectorXd, int> gradients(const VectorXd& y) const override; 
    VectorXd get_parameters() const override; 
    int set_parameters(const VectorXd& params) override; 
};