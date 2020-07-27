#include <vector>
#include <eigen3/Eigen/Core>

using std::vector;
typedef Eigen::MatrixXd MatrixXd;
typedef Eigen::VectorXd VectorXd;
static const double pts[] = {(-1/sqrt(3)+1.0)/2.0, (1/sqrt(3)+1.0)/2.0};

void FemLaplaceTerm_forward(double *out, const double *u, const double *kappa, int m, int n, double h){
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

  for(int i = 0; i<n;i++){
    for(int j = 0; j<m;j++){
      Eigen::Vector4i idx;
      idx << i*(m+1) + j, i*(m+1)+j+1, (i+1)*(m+1)+j, (i+1)*(m+1)+j+1;
      Eigen::Vector4d uA;
      uA << u[idx[0]], u[idx[1]], u[idx[2]], u[idx[3]];
      Eigen::Vector4d local_vec;
      local_vec.setZero();
      for(int q = 0; q<2;q++){
        for(int p = 0; p<2;p++){
          int k = q*2+p;
          local_vec += B[k].transpose() * B[k] * uA * h * h * 0.25 * kappa[k + 4*(i*m+j)];
        }
      }
      for(int l = 0; l<4;l++)
        out[idx[l]] += local_vec[l]; 
    }
  }
}



void FemLaplaceTerm_backward(
  double * grad_u, double *grad_kappa, 
  const double * grad_out, 
  const double *out, const double *u, const double *kappa, int m, int n, double h){
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

  for(int i = 0; i<n;i++){
    for(int j = 0; j<m;j++){
      Eigen::Vector4i idx;
      idx << i*(m+1) + j, i*(m+1)+j+1, (i+1)*(m+1)+j, (i+1)*(m+1)+j+1;
      Eigen::Vector4d uA;
      uA << u[idx[0]], u[idx[1]], u[idx[2]], u[idx[3]];
      Eigen::Vector4d grad_oA;
      grad_oA << grad_out[idx[0]], grad_out[idx[1]], grad_out[idx[2]], grad_out[idx[3]];

      Eigen::Vector4d local_vec;
      local_vec.setZero();
      for(int q = 0; q<2;q++){
        for(int p = 0; p<2;p++){
          int k = q*2+p;
          grad_kappa[k +4*(i*m+j)] = grad_oA.dot( B[k].transpose() * B[k] * uA * h * h * 0.25 );
          local_vec = B[k].transpose() * B[k] * grad_oA * h * h * 0.25 * kappa[k + 4*(i*m+j)];
          for (int l = 0; l<4; l++) grad_u[idx[l]] += local_vec[l];
        }
      }
    }
  }
}