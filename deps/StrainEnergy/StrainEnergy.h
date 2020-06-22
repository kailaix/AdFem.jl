#include <vector>
#include <eigen3/Eigen/Core>

using std::vector;
static const double pts[] = {(-1/sqrt(3)+1.0)/2.0, (1/sqrt(3)+1.0)/2.0};


void SE_forward(double *out, const double *S, int m, int n, double h){
      Eigen::Matrix<double,3,8> Bs[4];
      Eigen::Matrix<double,1,3> s;
      Eigen::Vector<int,8> dof;

      for(int i=0;i<2*(m+1)*(n+1);i++) out[i] = 0.0;

      int k =0;
      for(int i=0;i<2;i++)
        for(int j=0;j<2;j++){
          double xi = pts[i], eta = pts[j];
          Bs[k++] << -1/h*(1-eta), 1/h*(1-eta), -1/h*eta, 1/h*eta, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, -1/h*(1-xi), -1/h*xi, 1/h*(1-xi), 1/h*xi,
            -1/h*(1-xi), -1/h*xi, 1/h*(1-xi), 1/h*xi, -1/h*(1-eta), 1/h*(1-eta), -1/h*eta, 1/h*eta;
          
        }
      
      for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
          int elem = j*m + i;
          const double *sigma = S + 3*(4*elem);
          dof << j*(m+1) + i, j*(m+1)+i+1, (j+1)*(m+1)+i, (j+1)*(m+1)+i+1,
              j*(m+1) + i + (m+1)*(n+1), j*(m+1)+i+1 + (m+1)*(n+1), (j+1)*(m+1)+i + (m+1)*(n+1), (j+1)*(m+1)+i+1 + (m+1)*(n+1);

          for(int p=0;p<2;p++){
            for(int q=0;q<2;q++){
              int idx = 2*q + p;
              
              s << sigma[3*idx], sigma[3*idx+1], sigma[3*idx+2];
              auto x = s * Bs[idx];
              for(int l = 0; l<8;l++){
                out[dof[l]] += x[l] * 0.25 * h * h;
              }
            }
          }
        }
      }

}

void SE_backward(
  double * grad_S, const double * grad_out, 
  const double *out, const double *S, int m, int n, double h
){
      Eigen::Matrix<double,3,8> Bs[4];
      Eigen::Matrix<double,1,3> s;
      Eigen::Vector<int,8> dof;
      Eigen::Vector<double, 8> g;


      int k =0;
      for(int i=0;i<2;i++)
        for(int j=0;j<2;j++){
          double xi = pts[i], eta = pts[j];
          Bs[k++] << -1/h*(1-eta), 1/h*(1-eta), -1/h*eta, 1/h*eta, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, -1/h*(1-xi), -1/h*xi, 1/h*(1-xi), 1/h*xi,
            -1/h*(1-xi), -1/h*xi, 1/h*(1-xi), 1/h*xi, -1/h*(1-eta), 1/h*(1-eta), -1/h*eta, 1/h*eta;
          
        }
      
      for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
          int elem = j*m + i;
          double *sigma = grad_S + 3*(4*elem);
          dof << j*(m+1) + i, j*(m+1)+i+1, (j+1)*(m+1)+i, (j+1)*(m+1)+i+1,
              j*(m+1) + i + (m+1)*(n+1), j*(m+1)+i+1 + (m+1)*(n+1), (j+1)*(m+1)+i + (m+1)*(n+1), (j+1)*(m+1)+i+1 + (m+1)*(n+1);

          for(int p=0;p<2;p++){
            for(int q=0;q<2;q++){
              int idx = 2*q + p;
              for(int l=0;l<8;l++) g[l] = grad_out[dof[l]];
              auto x = Bs[idx] * g * 0.25 * h * h;
              sigma[3*idx] = x[0]; 
              sigma[3*idx+1] = x[1];
              sigma[3*idx+2] = x[2];
            }
          }
        }
      }
}