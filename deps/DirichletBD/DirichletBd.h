#include <set>
#include <vector>
using std::set;
using std::vector;

class Forward{
  private:
    vector<int> ii1;
    vector<int> jj1;
    vector<double> vv1;
    vector<int> ii2;
    vector<int> jj2;
    vector<double> vv2;
  public:
    Forward(const int64 * ii, const int64 * jj, const double * vv, int N, 
      const int*bd, int bdn, int m, int n, double h){
        set<int> bdset(bd, bd+bdn);
        for(int i=0;i<bdn;i++) bdset.insert(bd[i]+(m+1)*(n+1));
        for(int i=0;i<N;i++){
          if(bdset.count(ii[i])>0 || bdset.count(jj[i])>0) continue;
          ii1.push_back(ii[i]); jj1.push_back(jj[i]); vv1.push_back(vv[i]); 
          if(bdset.count(jj[i])>0 && bdset.count(ii[i])==0){
              ii2.push_back(ii[i]); jj2.push_back(jj[i]); vv2.push_back(vv[i]); 
          }
        }
        for(auto i: bdset) {
          ii1.push_back(i); jj1.push_back(i); vv1.push_back(1.0); 
        }
    }

    void fill(OpKernelContext* context){
      int n1 = ii1.size(), n2 = ii2.size();

      TensorShape ii1_shape({n1});
      TensorShape jj1_shape({n1});
      TensorShape vv1_shape({n1});
      TensorShape ii2_shape({n2});
      TensorShape jj2_shape({n2});
      TensorShape vv2_shape({n2});
              
      // create output tensor
      
      Tensor* ii1__ = NULL;
      OP_REQUIRES_OK(context, context->allocate_output(0, ii1_shape, &ii1__));
      Tensor* jj1__ = NULL;
      OP_REQUIRES_OK(context, context->allocate_output(1, jj1_shape, &jj1__));
      Tensor* vv1__ = NULL;
      OP_REQUIRES_OK(context, context->allocate_output(2, vv1_shape, &vv1__));
      Tensor* ii2__ = NULL;
      OP_REQUIRES_OK(context, context->allocate_output(3, ii2_shape, &ii2__));
      Tensor* jj2__ = NULL;
      OP_REQUIRES_OK(context, context->allocate_output(4, jj2_shape, &jj2__));
      Tensor* vv2__ = NULL;
      OP_REQUIRES_OK(context, context->allocate_output(5, vv2_shape, &vv2__));

      auto ii1_ = ii1__->flat<int64>().data();
      auto jj1_ = jj1__->flat<int64>().data();
      auto vv1_ = vv1__->flat<double>().data();
      auto ii2_ = ii2__->flat<int64>().data();
      auto jj2_ = jj2__->flat<int64>().data();
      auto vv2_ = vv2__->flat<double>().data();   
        
      for(int i=0;i<n1;i++){
        ii1_[i] = ii1[i];
        jj1_[i] = jj1[i];
        vv1_[i] = vv1[i];
      }
      for(int i=0;i<n2;i++){
        ii2_[i] = ii2[i];
        jj2_[i] = jj2[i];
        vv2_[i] = vv2[i];
      }
  }
};

void backward(
  double * grad_vv, const int64 * ii, const int64 * jj,
  const double * grad_vv1, const double * grad_vv2, int N, 
      const int*bd, int bdn, int m, int n, double h){
    set<int> bdset(bd, bd+bdn);
    for(int i=0;i<N;i++) grad_vv[i] = 0.0;
    for(int i=0;i<bdn;i++) bdset.insert(bd[i]+(m+1)*(n+1));


    int k1 = 0, k2 = 0;
    for(int i=0;i<N;i++){
      if(bdset.count(ii[i])>0 || bdset.count(jj[i])>0) continue;
      grad_vv[i] += grad_vv1[k1++];
      if(bdset.count(jj[i])>0 && bdset.count(ii[i])==0){
          grad_vv[i] += grad_vv2[k2++];
      }
    }
}