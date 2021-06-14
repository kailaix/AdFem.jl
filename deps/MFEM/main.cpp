#include "Common.h"

int main(){
    Eigen::VectorXd xs, w;
    line_integral_gauss_quadrature(xs, w, 6);
    std::cout << xs << std::endl;
    std::cout << w << std::endl;
    return 1;
}