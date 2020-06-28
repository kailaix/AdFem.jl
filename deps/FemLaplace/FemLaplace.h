#include <vector>
#include <eigen3/Eigen/Core>

using std::vector;
typedef Eigen::MatrixXd MatrixXd;
typedef Eigen::VectorXd VectorXd;

static const double pts[] = {(-1/sqrt(3)+1.0)/2.0, (1/sqrt(3)+1.0)/2.0};


void FemLaplace_forward(int64 *ii, int64 *jj, double *vv, 
  const double *K, int m, int n, double h){
    vector<MatrixXd> B; 
    for(int q = 0; q<2; q++){
      for(int p=0;p<2;p++){
        int k = q*2 + p;
        double xi = pts[p], eta = pts[q];
        MatrixXd B0(2, 4);
        B0 << -1/h*(1-eta), 1/h*(1-eta), -1/h*eta, 1/h*eta,
                  -1/h*(1-xi), -1/h*xi, 1/h*(1-xi), 1/h*xi;
        B.push_back(B0.transpose() * B0 * 0.25 * h * h);
      }
    }

    int k_gauss = 0;
    int k = 0;
    for(int j = 0; j<n;j++){
      for(int i = 0; i<m;i++){
        Eigen::VectorXi idx(4);
        idx << j*(m+1)+i, j*(m+1)+i+1, (j+1)*(m+1)+i, (j+1)*(m+1)+i+1;  
        for(int q = 0; q<2;q ++){
          for(int p = 0;p<2;p++){
            MatrixXd B0 = B[2*q+p];
            MatrixXd Omega = K[k_gauss++] * B0;
            for(int i_ = 0;i_<4;i_++){
              for(int j_=0;j_<4;j_++){
                ii[k] = idx[i_];
                jj[k] = idx[j_];
                vv[k] = Omega(i_, j_);
                k += 1;
              }
            }
          }
        }
      }
    }
}

void FemLaplace_backward(double *grad_K, 
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
        B.push_back(B0.transpose() * B0 * 0.25 * h * h);
      }
    }

    int k_gauss = 0;
    int k = 0;
    for(int j = 0; j<n;j++){
      for(int i = 0; i<m;i++){
        for(int q = 0; q<2;q ++){
          for(int p = 0;p<2;p++){
            MatrixXd B0 = B[2*q+p];
            for(int i_ = 0;i_<4;i_++){
              for(int j_=0;j_<4;j_++){
                grad_K[k_gauss] += B0(i_, j_) * grad_vv[k];
                k += 1;
              }
            }
            k_gauss ++;
          }
        }
      }
    }
}