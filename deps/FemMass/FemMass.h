#include <vector>
#include <eigen3/Eigen/Core>
static const double pts[] = {(-1/sqrt(3)+1.0)/2.0, (1/sqrt(3)+1.0)/2.0};

using std::vector;
typedef Eigen::MatrixXd MatrixXd;
typedef Eigen::VectorXd VectorXd;


void FemMass_forward(int64 *ii, int64 *jj, double *vv, 
  const double *rho, int m, int n, double h){
  vector<MatrixXd> Me(4);
  for (int q = 0; q < 2; q++){
    for(int p = 0; p < 2; p++){
      double xi = pts[p], eta = pts[q];
      MatrixXd A(4,1);
      A << (1-xi)*(1-eta), xi*(1-eta), (1-xi)*eta, xi*eta;
      Me[q*2 + p] = A  * A.transpose() * 0.25 * h*h;
    }
  }

  int k = 0;
  for(int j =0;j<n;j++){
    for(int i = 0; i< m;i++){
      int elem_idx = j*m+i;
      Eigen::VectorXi idx(4);
      idx << j*(m+1)+i, j*(m+1)+i+1, (j+1)*(m+1)+i, (j+1)*(m+1)+i+1;
      for(int q=0;q<2;q++)
        for(int p=0;p<2;p++){
          double rho_ = rho[elem_idx*4 + 2*q+ p];
          for(int i_=0;i_<4;i_++)
            for(int j_=0;j_<4;j_++){
              ii[k] = idx[i_];
              jj[k] = idx[j_];
              vv[k] = rho_ * Me[2*q+p](i_, j_);
              k++;
            }
        }
    }
  }

}

void FemMass_backward(double *grad_rho, const double *grad_vv, 
  int m, int n, double h){
    vector<MatrixXd> Me(4);
    for (int q = 0; q < 2; q++){
      for(int p = 0; p < 2; p++){
        double xi = pts[p], eta = pts[q];
        MatrixXd A(4,1);
        A << (1-xi)*(1-eta), xi*(1-eta), (1-xi)*eta, xi*eta;
        Me[q*2 + p] = A  * A.transpose() * 0.25 * h*h;
      }
    }

    int k = 0;
    for(int j =0;j<n;j++){
      for(int i = 0; i< m;i++){
        int elem_idx = j*m+i;
        Eigen::VectorXi idx(4);
        // idx << j*(m+1)+i, j*(m+1)+i+1, (j+1)*(m+1)+i, (j+1)*(m+1)+i+1;
        for(int q=0;q<2;q++)
          for(int p=0;p<2;p++){
            // double rho_ = rho[elem_idx*4 + 2*q+ p];
            
            for(int i_=0;i_<4;i_++)
              for(int j_=0;j_<4;j_++){
                // ii[k] = idx[i_];
                // jj[k] = idx[j_];
                // vv[k] = rho_ * Me[2*q+p](i_, j_);
                grad_rho[elem_idx*4 + 2*q+ p] += grad_vv[k] * Me[2*q+p](i_, j_);
                k++;
              }
          }
      }
    }

    
}

