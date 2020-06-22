#include <vector>
#include <eigen3/Eigen/Core>

using std::vector;
static const double pts[] = {(-1/sqrt(3)+1.0)/2.0, (1/sqrt(3)+1.0)/2.0};

class Forward{
  private:
    vector<int> ii, jj;
    vector<double> vv;
  public:
    Forward(const double *hmat, int m, int n, double h){
      Eigen::Matrix<double,8,8> Omega;
      Eigen::Matrix<double,3,3> K;
      Eigen::Vector<int,8> idx;
      Eigen::Matrix<double,3,8> Bs[4];

      int k =0;
      for(int i=0;i<2;i++)
        for(int j=0;j<2;j++){
          double xi = pts[i], eta = pts[j];
          Bs[k++] << -1/h*(1-eta), 1/h*(1-eta), -1/h*eta, 1/h*eta, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, -1/h*(1-xi), -1/h*xi, 1/h*(1-xi), 1/h*xi,
            -1/h*(1-xi), -1/h*xi, 1/h*(1-xi), 1/h*xi, -1/h*(1-eta), 1/h*(1-eta), -1/h*eta, 1/h*eta;
          
        }

      // std::cout << Omega << std::endl;
      for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
          int elem = j*m + i; 
          idx << j*(m+1)+i, j*(m+1)+i+1, (j+1)*(m+1)+i, (j+1)*(m+1)+i+1,
               j*(m+1)+i + (m+1)*(n+1), j*(m+1)+i+1 + (m+1)*(n+1), (j+1)*(m+1)+i + (m+1)*(n+1), (j+1)*(m+1)+i+1 + (m+1)*(n+1);
          for(int p=0;p<2;p++)
            for(int q=0;q<2;q++){
              int k = 2*q + p; 
              double xi = pts[p], eta = pts[q];
              K << hmat[36*elem+9*k], hmat[36*elem+9*k+1], hmat[36*elem+9*k+2],
                    hmat[36*elem+9*k+3], hmat[36*elem+9*k+4], hmat[36*elem+9*k+5],
                    hmat[36*elem+9*k+6], hmat[36*elem+9*k+7], hmat[36*elem+9*k+8];
              Omega = Bs[k].transpose() * K * Bs[k] * 0.25 * h* h;

              for(int r=0;r<8;r++){
                  for(int s=0;s<8;s++){
                    ii.push_back(idx[r]+1);
                    jj.push_back(idx[s]+1);
                    vv.push_back(Omega(r,s));
                  }
                }
            }

        }
      }
    }

    void fill(OpKernelContext* context){
      int N = ii.size();
      TensorShape ii_shape({N});
      TensorShape jj_shape({N});
      TensorShape vv_shape({N});
              
      // create output tensor
      
      Tensor* ii_ = NULL;
      OP_REQUIRES_OK(context, context->allocate_output(0, ii_shape, &ii_));
      Tensor* jj_ = NULL;
      OP_REQUIRES_OK(context, context->allocate_output(1, jj_shape, &jj_));
      Tensor* vv_ = NULL;
      OP_REQUIRES_OK(context, context->allocate_output(2, vv_shape, &vv_));
      
      // get the corresponding Eigen tensors for data access
      auto ii_tensor = ii_->flat<int64>().data();
      auto jj_tensor = jj_->flat<int64>().data();
      auto vv_tensor = vv_->flat<double>().data(); 
      for(int i=0;i<N;i++){
        ii_tensor[i] = ii[i];
        jj_tensor[i] = jj[i];
        vv_tensor[i] = vv[i];
      }
    }
};



void SFS_backward(
  double *grad_hmat, const double * grad_vv,  
  int m, int n, double h
){
      Eigen::Matrix<double,8, 8> dOmega;
      Eigen::Matrix<double,3,3> dK;
      Eigen::Vector<int,8> idx;
      Eigen::Matrix<double,3,8> Bs[4];

      int k =0;
      for(int i=0;i<2;i++)
        for(int j=0;j<2;j++){
          double xi = pts[i], eta = pts[j];
          Bs[k++] << -1/h*(1-eta), 1/h*(1-eta), -1/h*eta, 1/h*eta, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, -1/h*(1-xi), -1/h*xi, 1/h*(1-xi), 1/h*xi,
            -1/h*(1-xi), -1/h*xi, 1/h*(1-xi), 1/h*xi, -1/h*(1-eta), 1/h*(1-eta), -1/h*eta, 1/h*eta;
          
        }

      // std::cout << Omega << std::endl;
      int rs_ = 0;
      for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
          int elem = j*m + i; 
          idx << j*(m+1)+i, j*(m+1)+i+1, (j+1)*(m+1)+i, (j+1)*(m+1)+i+1,
               j*(m+1)+i + (m+1)*(n+1), j*(m+1)+i+1 + (m+1)*(n+1), (j+1)*(m+1)+i + (m+1)*(n+1), (j+1)*(m+1)+i+1 + (m+1)*(n+1);
          
              
          for(int p=0;p<2;p++)
            for(int q=0;q<2;q++){
              int k = 2*q + p;               
              for(int r = 0; r<8; r++)
                for(int s = 0; s<8; s++){
                  dOmega(s, r) = grad_vv[rs_++];
                }
              dK = Bs[k] * dOmega * Bs[k].transpose() * 0.25 * h* h;
              // std::cout << "Omega\n" << dOmega << std::endl;
              // std::cout << dK << std::endl;
              int rs = 0;
              for(int r = 0; r<3; r++)
                for(int s = 0; s<3; s++){
                  grad_hmat[36*elem+9*k+rs] = dK(s, r);
                  rs++;
                }

            }

        }
      }
    };