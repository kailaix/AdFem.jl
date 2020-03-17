#include <vector>
#include <eigen3/Eigen/Core>

using std::vector;
static const double pts[] = {(-1/sqrt(3)+1.0)/2.0, (1/sqrt(3)+1.0)/2.0};


void forward(double *strain, const double *u, int m, int n, double h){
      Eigen::Matrix<double,3,8> Bs[4];
      Eigen::Vector<int,8> dof;
      Eigen::Vector<double,8> uA;

      for(int i=0;i<2;i++)
        for(int j=0;j<2;j++){
          int idx = 2*j + i;
          double xi = pts[i], eta = pts[j];
          Bs[idx] << -1/h*(1-eta), 1/h*(1-eta), -1/h*eta, 1/h*eta, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, -1/h*(1-xi), -1/h*xi, 1/h*(1-xi), 1/h*xi,
            -1/h*(1-xi), -1/h*xi, 1/h*(1-xi), 1/h*xi, -1/h*(1-eta), 1/h*(1-eta), -1/h*eta, 1/h*eta;
          
        }
      
      for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
          int elem = j*m + i;
          dof << j*(m+1) + i, j*(m+1)+i+1, (j+1)*(m+1)+i, (j+1)*(m+1)+i+1,
              j*(m+1) + i + (m+1)*(n+1), j*(m+1)+i+1 + (m+1)*(n+1), (j+1)*(m+1)+i + (m+1)*(n+1), (j+1)*(m+1)+i+1 + (m+1)*(n+1);
          for(int l=0;l<8;l++) uA[l] = u[dof[l]];
          strain[elem] = 0.0;
          for(int p=0;p<2;p++){
            for(int q=0;q<2;q++){
              int idx = 2*q + p;
              auto x = Bs[idx] * uA * h * h  * 0.25;
              strain[elem] += x[0] + x[1];
            }
          }
        }
      }

}

void backward(
  double * grad_u, const double * grad_strain, 
  const double *strain, const double *u, int m, int n, double h
){
      Eigen::Matrix<double,3,8> Bs[4];
      Eigen::Matrix<double,1,3> gs;
      Eigen::Vector<int,8> dof;

      for(int i=0;i<2*(m+1)*(n+1);i++) grad_u[i] = 0.0;
      Eigen::Matrix<double,1,3> v;
      v << 1.0, 1.0, 0.0;

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
          dof << j*(m+1) + i, j*(m+1)+i+1, (j+1)*(m+1)+i, (j+1)*(m+1)+i+1,
              j*(m+1) + i + (m+1)*(n+1), j*(m+1)+i+1 + (m+1)*(n+1), (j+1)*(m+1)+i + (m+1)*(n+1), (j+1)*(m+1)+i+1 + (m+1)*(n+1);
          for(int p=0;p<2;p++){
            for(int q=0;q<2;q++){
              int idx = 2*q + p;
              auto g = v * Bs[idx] * h * h * 0.25;
              for (int l=0;l<8;l++)
                  grad_u[dof[l]] += g(0,l)*grad_strain[elem];
            }
          }
        }
      }
}