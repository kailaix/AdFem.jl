#include <vector>
#include <eigen3/Eigen/Core>

using std::vector;
static const double pts[] = {(-1/sqrt(3)+1.0)/2.0, (1/sqrt(3)+1.0)/2.0};


void FemSource_forward(double *rhs, const double *f, int m, int n, double h){
    for(int i=0;i<m;i++){
      for(int j = 0; j<n;j++){
        int idx = j*m + i;
        for(int p = 0;p<2;p++){
          for(int q = 0;q<2;q++){
            double xi = pts[p], eta = pts[q];
            int k = idx * 4 + 2*q + p;
            double val1 = f[k] * h*h*0.25;
            rhs[j*(m+1)+i] += val1 * (1-xi) * (1-eta);
            rhs[j*(m+1)+i+1] += val1 * xi * (1-eta);
            rhs[(j+1)*(m+1)+i] += val1 * (1-xi) * eta;
            rhs[(j+1)*(m+1)+i+1] += val1 * xi * eta;
          }
        }
      }
    }
}



void FemSource_backward(
  double *grad_f, 
  const double *grad_rhs, int m, int n, double h){
    for(int i=0;i<m;i++){
      for(int j = 0; j<n;j++){
        int idx = j*m + i;
        for(int p = 0;p<2;p++){
          for(int q = 0;q<2;q++){
            double xi = pts[p], eta = pts[q];
            int k = idx * 4 + 2*q + p;
            grad_f[k] += h*h*0.25*(1-xi) * (1-eta) * grad_rhs[j*(m+1)+i];
            grad_f[k] += h*h*0.25*xi * (1-eta) * grad_rhs[j*(m+1)+i+1];
            grad_f[k] += h*h*0.25*(1-xi) * eta * grad_rhs[(j+1)*(m+1)+i];
            grad_f[k] += h*h*0.25*xi * eta * grad_rhs[(j+1)*(m+1)+i+1];
          }
        }
      }
    }
}