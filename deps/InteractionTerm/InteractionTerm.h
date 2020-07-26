#include <vector>
#include <eigen3/Eigen/Core>

using std::vector;
typedef Eigen::MatrixXd MatrixXd;
typedef Eigen::VectorXd VectorXd;

static const double pts[] = {(-1/sqrt(3)+1.0)/2.0, (1/sqrt(3)+1.0)/2.0};


void InteractionTerm_forward(
  double *out, const double *pp, int m, int n, double h
){
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

// p dudx, p dvdy
// out 2(m+1)*(n+1)
  for(int j=0;j<n;j++){
    for(int i = 0; i<m; i++){
      double p0 = pp[j*m+i];
      int vidx[] = {j*(m+1)+i, j*(m+1)+i+1, (j+1)*(m+1)+i, (j+1)*(m+1)+i+1};
      for(int q = 0; q<2;q++){
        for(int p = 0;p<2;p++){
          int idx = p + 2*q;
          MatrixXd B0 = B[idx];
          for(int l = 0; l<4;l++){
            out[vidx[l]] += B0(0,l) * p0 * h * h * 0.25;
            out[vidx[l]+(m+1)*(n+1)] += B0(1,l) * p0 * h * h * 0.25;
          }
        }
      }
    }
  }
}


void InteractionTerm_backward(
  double * grad_p, const double * grad_out,
  const double *out, const double *pp, int m, int n, double h
){
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

// p dudx, p dvdy
// out 2(m+1)*(n+1)
  for(int j=0;j<n;j++){
    for(int i = 0; i<m; i++){
      double *grad_p0 = grad_p+j*m+i;
      int vidx[] = {j*(m+1)+i, j*(m+1)+i+1, (j+1)*(m+1)+i, (j+1)*(m+1)+i+1};
      for(int q = 0; q<2;q++){
        for(int p = 0;p<2;p++){
          int idx = p + 2*q;
          MatrixXd B0 = B[idx];
          for(int l = 0; l<4;l++){
            *grad_p0 += grad_out[vidx[l]] * B0(0,l) * h * h * 0.25;
            *grad_p0 += grad_out[vidx[l]+(m+1)*(n+1)] * B0(1,l) * h * h * 0.25;
          }
        }
      }
    }
  }
}