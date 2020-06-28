#include <vector>
#include <eigen3/Eigen/Core>
static const double pts[] = {(-1/sqrt(3)+1.0)/2.0, (1/sqrt(3)+1.0)/2.0};

using std::vector;
typedef Eigen::MatrixXd MatrixXd;
typedef Eigen::VectorXd VectorXd;


void FemToGaussPoints_forward(double *ret, 
  const double *u, int m, int n, double h){
  vector<VectorXd> Me(4);
  for (int q = 0; q < 2; q++){
    for(int p = 0; p < 2; p++){
      double xi = pts[p], eta = pts[q];
      VectorXd A(4);
      A << (1-xi)*(1-eta), xi*(1-eta), (1-xi)*eta, xi*eta;
      Me[q*2 + p] = A;
    }
  }

  int k = 0;
  for(int j =0;j<n;j++){
    for(int i = 0; i< m;i++){
      int elem_idx = j*m+i;
      for(int q=0;q<2;q++)
        for(int p=0;p<2;p++){
          ret[elem_idx*4 + 2*q + p] = Me[q*2+p][0] * u[j*(m+1)+i] + \
                                      Me[q*2+p][1] * u[j*(m+1)+i+1] + \
                                      Me[q*2+p][2] * u[(j+1)*(m+1)+i] + \
                                      Me[q*2+p][3] * u[(j+1)*(m+1)+i+1];
        }
    }
  }

}


void FemToGaussPoints_backward(
  double *grad_u,
  const double *grad_ret, 
  int m, int n, double h){
  vector<VectorXd> Me(4);
  for (int q = 0; q < 2; q++){
    for(int p = 0; p < 2; p++){
      double xi = pts[p], eta = pts[q];
      VectorXd A(4);
      A << (1-xi)*(1-eta), xi*(1-eta), (1-xi)*eta, xi*eta;
      Me[q*2 + p] = A;
    }
  }

  int k = 0;
  for(int j =0;j<n;j++){
    for(int i = 0; i< m;i++){
      int elem_idx = j*m+i;
      for(int q=0;q<2;q++)
        for(int p=0;p<2;p++){
          grad_u[j*(m+1)+i] += Me[q*2+p][0] * grad_ret[elem_idx*4 + 2*q + p];
          grad_u[j*(m+1)+i+1] += Me[q*2+p][1] * grad_ret[elem_idx*4 + 2*q + p];
          grad_u[(j+1)*(m+1)+i] += Me[q*2+p][2] * grad_ret[elem_idx*4 + 2*q + p];
          grad_u[(j+1)*(m+1)+i+1] += Me[q*2+p][3] * grad_ret[elem_idx*4 + 2*q + p];
        }
    }
  }

}
