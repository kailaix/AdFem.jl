#include <vector>
#include <eigen3/Eigen/Core>

using std::vector;
static const double pts[] = {(-1/sqrt(3)+1.0)/2.0, (1/sqrt(3)+1.0)/2.0};


void SO1_forward(double *strain, const double *u, int m, int n, double h){
      Eigen::Matrix<double,2, 4> Bs[4];
      Eigen::Vector<int,4> dof;
      Eigen::Vector<double,4> uA;

      for(int i=0;i<2;i++)
        for(int j=0;j<2;j++){
          double xi = pts[i], eta = pts[j];
          int idx = 2*j + i;
          Bs[idx] << -1/h*(1-eta), 1/h*(1-eta), -1/h*eta, 1/h*eta,
                -1/h*(1-xi), -1/h*xi, 1/h*(1-xi), 1/h*xi;
          
        }
      
      for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
          int elem = j*m + i;
          dof << j*(m+1) + i, j*(m+1)+i+1, (j+1)*(m+1)+i, (j+1)*(m+1)+i+1;
          for(int l=0;l<4;l++) uA[l] = u[dof[l]];
          for(int p=0;p<2;p++){
            for(int q=0;q<2;q++){
              int idx = 2*q + p;
              Eigen::VectorXd x = Bs[idx] * uA;
              // std::cout << "***\n" <<  Bs[idx] << std::endl  << std::endl;
              strain[2*(4*elem + idx)] = x[0];
              strain[2*(4*elem + idx)+1] = x[1];
            }
          }
        }
      }

}

void SO1_backward(
  double * grad_u, const double * grad_strain, 
  const double *strain, const double *u, int m, int n, double h
){
      Eigen::Matrix<double,2, 4> Bs[4];
      Eigen::Matrix<double,1,2> gs;
      Eigen::Vector<int,4> dof;

      for(int i=0;i<(m+1)*(n+1);i++) grad_u[i] = 0.0;

      for(int i=0;i<2;i++)
        for(int j=0;j<2;j++){
          double xi = pts[i], eta = pts[j];
          int idx = 2*j + i;
          Bs[idx]  << -1/h*(1-eta), 1/h*(1-eta), -1/h*eta, 1/h*eta,
                -1/h*(1-xi), -1/h*xi, 1/h*(1-xi), 1/h*xi;
          
        }
      
      for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
          int elem = j*m + i;
          dof << j*(m+1) + i, j*(m+1)+i+1, (j+1)*(m+1)+i, (j+1)*(m+1)+i+1;
          for(int p=0;p<2;p++){
            for(int q=0;q<2;q++){
              int idx = 2*q + p;
              gs <<   grad_strain[2*(4*elem + idx)],
                      grad_strain[2*(4*elem + idx)+1];
              Eigen::MatrixXd x = gs * Bs[idx]; // 1 x 8 vector
              for(int l=0;l<4;l++) grad_u[dof[l]] += x(0, l);
            }
          }
        }
      }
}