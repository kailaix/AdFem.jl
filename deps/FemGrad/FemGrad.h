#include <vector>
#include <eigen3/Eigen/Core>

using std::vector;
typedef Eigen::MatrixXd MatrixXd;
typedef Eigen::VectorXd VectorXd;

static const double pts[] = {(-1/sqrt(3)+1.0)/2.0, (1/sqrt(3)+1.0)/2.0};

/* ret 4mn x 2 

[
  o_11 o_12
  o_21 o_22
  o_31 o_32
  ...
]

--> C++ memory

o_11, o_12, o_21, o_22, ...

*/
void FemGrad_forward(double *ret, const double *u, int m, int n, double h){
  vector<MatrixXd> B; 
  for(int q = 0; q<2; q++){
    for(int p=0;p<2;p++){
      int k = q*2 + p;
      double xi = pts[p], eta = pts[q];
      MatrixXd B0(2, 4);
      B0 << -1/h*(1-eta), 1/h*(1-eta), -1/h*eta, 1/h*eta,
                -1/h*(1-xi), -1/h*xi, 1/h*(1-xi), 1/h*xi;
      B.push_back(B0);
    }
  }
  int ret_idx = 0;
  for(int j=0;j<n;j++){
    for(int i=0; i<m;i++){
      VectorXd uA(4);
      uA << u[j*(m+1)+i], u[j*(m+1)+i+1], u[(j+1)*(m+1)+i], u[(j+1)*(m+1)+i+1];
      for(int q = 0; q<2; q++){
        for(int p = 0;p <2;p++){
          int k = q*2+p;
          VectorXd g = B[k] * uA;
          ret[ret_idx++] = g[0];
          ret[ret_idx++] = g[1]; 
        }
      }
    }
  }
}


void FemGrad_backward(double *grad_u, const double *grad_ret, 
  const double*u, int m, int n, double h){
  vector<MatrixXd> B; 
  for(int q = 0; q<2; q++){
    for(int p=0;p<2;p++){
      int k = q*2 + p;
      double xi = pts[p], eta = pts[q];
      MatrixXd B0(2, 4);
      B0 << -1/h*(1-eta), 1/h*(1-eta), -1/h*eta, 1/h*eta,
                -1/h*(1-xi), -1/h*xi, 1/h*(1-xi), 1/h*xi;
      B.push_back(B0.transpose());
    }
  }
  int ret_idx = 0;
  for(int j=0;j<n;j++){
    for(int i=0; i<m;i++){
      VectorXd uA(4);
      uA << u[j*(m+1)+i], u[j*(m+1)+i+1], u[(j+1)*(m+1)+i], u[(j+1)*(m+1)+i+1];
      for(int q = 0; q<2; q++){
        for(int p = 0;p <2;p++){
          int k = q*2+p;
          VectorXd grad_ret_0(2);
          grad_ret_0 << grad_ret[ret_idx], grad_ret[ret_idx+1];
          VectorXd g0 = B[k] * grad_ret_0;
          grad_u[j*(m+1)+i] += g0[0];
          grad_u[j*(m+1)+i+1] += g0[1];
          grad_u[(j+1)*(m+1)+i] += g0[2];
          grad_u[(j+1)*(m+1)+i+1] += g0[3];
          ret_idx += 2;
        }
      }
    }
  }
}

