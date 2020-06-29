#include <vector>
#include <eigen3/Eigen/Core>

using std::vector;
typedef Eigen::MatrixXd MatrixXd;
typedef Eigen::VectorXd VectorXd;

static const double pts[] = {(-1/sqrt(3)+1.0)/2.0, (1/sqrt(3)+1.0)/2.0};


void FemAdvection_forward(int64 *ii, int64*jj, double *vv, 
  const double *u0, const double *v0, int m, int n, double h){
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

  vector<MatrixXd> Me(4);
  for (int q = 0; q < 2; q++){
    for(int p = 0; p < 2; p++){
      double xi = pts[p], eta = pts[q];
      MatrixXd A(4,1);
      A << (1-xi)*(1-eta), xi*(1-eta), (1-xi)*eta, xi*eta;
      Me[q*2 + p] = A;
    }
  }

  int uv_k = 0;
  int k = 0;
  for(int j = 0; j < n; j++){
    for(int i = 0; i < m;i++){
      Eigen::VectorXi idx(4);
      idx << j*(m+1)+i, j*(m+1)+i+1, (j+1)*(m+1)+i, (j+1)*(m+1)+i+1;
      for(int q=0;q<2;q++){
        for(int p=0;p<2;p++){
          MatrixXd N = Me[q*2 + p];
          MatrixXd B0 = B[q*2 + p];
          MatrixXd uv(1,2);
          uv << u0[uv_k], v0[uv_k];
          uv_k+=1;
          MatrixXd R = N * uv * B0;
          for(int i_=0;i_<4;i_++)
            for(int j_=0;j_<4;j_++){
              ii[k] = idx[i_];
              jj[k] = idx[j_];
              vv[k] = R(i_, j_) * 0.25 * h *h;
              k++;
            }
        }
      }
    }
  }
}

void FemAdvection_backward(double *grad_u0, double *grad_v0, 
  const double *grad_vv, 
  int m, int n, double h){
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

      vector<MatrixXd> Me(4);
      for (int q = 0; q < 2; q++){
        for(int p = 0; p < 2; p++){
          double xi = pts[p], eta = pts[q];
          MatrixXd A(4,1);
          A << (1-xi)*(1-eta), xi*(1-eta), (1-xi)*eta, xi*eta;
          Me[q*2 + p] = A;
        }
      }

      int uv_k = 0;
      int k = 0;
      for(int j = 0; j < n; j++){
        for(int i = 0; i < m;i++){
          Eigen::VectorXi idx(4);
          idx << j*(m+1)+i, j*(m+1)+i+1, (j+1)*(m+1)+i, (j+1)*(m+1)+i+1;
          for(int q=0;q<2;q++){
            for(int p=0;p<2;p++){
              MatrixXd R(4,4);
              for(int i_=0;i_<4;i_++)
                for(int j_=0;j_<4;j_++){
                  R(i_, j_) = grad_vv[k] * 0.25 * h * h; 
                  k++;
                }
                MatrixXd N = Me[q*2 + p];
                MatrixXd B0 = B[q*2 + p];
                MatrixXd uv = N.transpose() * R * B0.transpose();
                grad_u0[uv_k] += uv(0, 0);
                grad_v0[uv_k] += uv(0, 1);
                uv_k += 1;
            }
          }
        }
      }
}