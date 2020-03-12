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
      Eigen::Matrix<double,4,4> Omega;
      Eigen::Matrix<double,2,4> B;
      Eigen::Matrix<double,2,2> K;
      Eigen::Vector<int,4> idx;
      Omega.setZero();
      for(int i=0;i<2;i++)
        for(int j=0;j<2;j++)
          K(i,j) = hmat[j*2+i];
      for(int i=0;i<2;i++)
        for(int j=0;j<2;j++){
          double xi = pts[i], eta = pts[j];
          B << -1/h*(1-eta), 1/h*(1-eta), -1/h*eta, 1/h*eta
                -1/h*(1-xi), -1/h*xi, 1/h*(1-xi), 1/h*xi;
          Omega += B.transpose() * K * B * 0.25 * h* h;
        }
      // std::cout << Omega << std::endl;
      for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
          idx << j*(m+1)+i, j*(m+1)+i+1, (j+1)*(m+1)+i, (j+1)*(m+1)+i+1;
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
      for(int i=0;i<9;i++) grad_hmat[i] = 0.0;
      int k = 0;
      for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
          dOmega.setZero();
          for(int p=0;p<4;p++){
            for(int q=0;q<4;q++){
              dOmega(p, q) = grad_vv[k++];
            }
          }

          dK.setZero();
          for(int i=0;i<2;i++)
            for(int j=0;j<2;j++){
              double xi = pts[i], eta = pts[j];
              B << -1/h*(1-eta), 1/h*(1-eta), -1/h*eta, 1/h*eta
                -1/h*(1-xi), -1/h*xi, 1/h*(1-xi), 1/h*xi;
              dK += B * dOmega * B.transpose() * 0.25 * h* h;
            }

          for(int i=0;i<2;i++){
            for(int j=0;j<2;j++){
              grad_hmat[j*2+i] += dK(i,j);
            }
          }
            
        }
      }
    };