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
      // printf("m=%d, n=%d\n", m, n); 
      Eigen::Matrix<double,4,4> Omega;
      Eigen::Matrix<double,2,4> Bs[4];
      Eigen::Matrix<double,4,2> Bt[4];
      Eigen::Matrix<double,2,2> K;
      Eigen::Vector<int,4> idx;

      for(int ei=0;ei<2;ei++)
            for(int ej=0;ej<2;ej++){
              double xi = pts[ei], eta = pts[ej];
              Bs[2*ej+ei] << -1/h*(1-eta), 1/h*(1-eta), -1/h*eta, 1/h*eta,
                    -1/h*(1-xi), -1/h*xi, 1/h*(1-xi), 1/h*xi;
              Bt[2*ej+ei] = Bs[2*ej+ei].transpose();
        }
      
      // std::cout << Omega << std::endl;
      for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
          idx << j*(m+1)+i, j*(m+1)+i+1, (j+1)*(m+1)+i, (j+1)*(m+1)+i+1;
          for(int ei=0;ei<2;ei++)
            for(int ej=0;ej<2;ej++){
              int ids = 16*(i+j*m) + 4*(ei+ej*2);
              K << hmat[ids], hmat[ids+1], 
                  hmat[ids+2], hmat[ids+3];
              // std::cout << "***\n" << K << std::endl;
              
              Omega = Bt[2*ej+ei] * K * Bs[2*ej+ei] * 0.25 * h* h;
              for(int p=0;p<4;p++){
                for(int q=0;q<4;q++){
                  ii.push_back(idx[p]+1);
                  jj.push_back(idx[q]+1);
                  vv.push_back(Omega(p,q));
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


void backward(
  double *grad_hmat, const double * grad_vv,  
  int m, int n, double h
){
      Eigen::Matrix<double,2,4> B;
      Eigen::Matrix<double,2,2> dK;
      Eigen::Matrix<double,4,4> dOmega;

      Eigen::Matrix<double,2,4> Bs[4];
      Eigen::Matrix<double,4,2> Bt[4];

      for(int ei=0;ei<2;ei++)
            for(int ej=0;ej<2;ej++){
              double xi = pts[ei], eta = pts[ej];
              Bs[2*ej+ei] << -1/h*(1-eta), 1/h*(1-eta), -1/h*eta, 1/h*eta,
                    -1/h*(1-xi), -1/h*xi, 1/h*(1-xi), 1/h*xi;
              Bt[2*ej+ei] = Bs[2*ej+ei].transpose();
        }

      int k = 0;
      for(int ei=0;ei<m;ei++){
        for(int ej=0;ej<n;ej++){
      
          for(int i=0;i<2;i++)
            for(int j=0;j<2;j++){
              int ids = 16*(ei+ej*m) + 4*(i+j*2);

              double xi = pts[i], eta = pts[j];
              for(int p=0;p<4;p++){
                for(int q=0;q<4;q++){
                  dOmega(p, q) = grad_vv[k++];
                }
              }
              dK = Bs[2*j+i] * dOmega * Bt[2*j+i] * 0.25 * h* h;

              // std::cout << dK << std::endl;
              grad_hmat[ids] = dK(0,0);
              grad_hmat[ids+1] = dK(0,1);
              grad_hmat[ids+2] = dK(1,0);
              grad_hmat[ids+3] = dK(1,1);

            }
        }
      }
    };