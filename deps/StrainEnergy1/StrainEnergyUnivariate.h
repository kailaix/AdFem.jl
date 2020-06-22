#include <vector>
#include <eigen3/Eigen/Core>
using std::vector;
static const double pts[] = {(-1/sqrt(3)+1.0)/2.0, (1/sqrt(3)+1.0)/2.0};

void SE1_forward(double *out, const double *stress, int m, int n ,double h){

    Eigen::MatrixXd stress_(1,2);
    Eigen::Matrix<double,2,4> Bs[4];
    Eigen::VectorXi idx(4);

    for(int ei=0;ei<2;ei++)
        for(int ej=0;ej<2;ej++){
          double xi = pts[ei], eta = pts[ej];
          Bs[ej*2+ei] << -1/h*(1-eta), 1/h*(1-eta), -1/h*eta, 1/h*eta,
                  -1/h*(1-xi), -1/h*xi, 1/h*(1-xi), 1/h*xi;
        }


    for(int i=0;i<(m+1)*(n+1);i++) out[i] = 0.0;

    for(int i=0;i<m;i++){
      for(int j=0;j<n;j++){
        idx << j*(m+1)+i, j*(m+1)+i+1, (j+1)*(m+1)+i, (j+1)*(m+1)+i+1;
        for(int ei=0;ei<2;ei++)
          for(int ej=0;ej<2;ej++){
            int elem = 4*2*(j*m+i) + (ei + 2*ej)*2;
            // std::cout << "***\n" << K << std::endl;
            stress_ << stress[elem], stress[elem+1];
            Eigen::MatrixXd x = stress_ * Bs[ej*2+ei] * 0.25 *h * h; // 1 x 4
            for (int r=0;r<4;r++){
              out[idx[r]] += x(0,r);
            }
            
          }
      }
    }
}

void SE1_backward(double *grad_stress, const double * grad_out,
  const double *out, const double *stress, int m, int n ,double h){
    Eigen::Matrix<double,2,4> Bs[4];
    Eigen::VectorXi idx(4);
    Eigen::VectorXd grad_(4);
    // for(int i=0;i<4*m*n;i++) grad_stress[i] = 0.0;
    for(int ei=0;ei<2;ei++)
        for(int ej=0;ej<2;ej++){
          double xi = pts[ei], eta = pts[ej];
          Bs[ej*2+ei] << -1/h*(1-eta), 1/h*(1-eta), -1/h*eta, 1/h*eta,
                  -1/h*(1-xi), -1/h*xi, 1/h*(1-xi), 1/h*xi;
        }

    for(int i=0;i<m;i++){
      for(int j=0;j<n;j++){
        idx << j*(m+1)+i, j*(m+1)+i+1, (j+1)*(m+1)+i, (j+1)*(m+1)+i+1;
        for(int ei=0;ei<2;ei++)
          for(int ej=0;ej<2;ej++){
            // std::cout << "***\n" << K << std::endl;
            double xi = pts[ei], eta = pts[ej];
            int elem = 4*2*(j*m+i) + (ei + 2*ej)*2;

            grad_ << grad_out[idx[0]], grad_out[idx[1]], grad_out[idx[2]], grad_out[idx[3]];
            auto x = Bs[ej*2+ei] * grad_* 0.25 *h * h; // vector of size 2
            grad_stress[elem] = x[0];
            grad_stress[elem+1] = x[1];
          }
      }
    }

}