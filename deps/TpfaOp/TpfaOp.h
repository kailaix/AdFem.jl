#include <map>
#include <vector>
#include <utility>
#include <algorithm>
using std::map;
using std::vector;
using std::pair;

void forward(
  OpKernelContext* context,
  double *rhs, 
  const double *K, const int64 *bc, const double *pval, int N, int m, int n, double h){
  map<pair<int,int>, double> bval;  
  for(int i=0;i<N;i++){
    int a = std::min(bc[2*i]-1, bc[2*i+1]-1);
    int b = std::max(bc[2*i]-1, bc[2*i+1]-1);
    bval[std::make_pair(a, b)] = pval[i];
    
  }
  vector<int> ivec, jvec;
  vector<double> vvec;
  for(int i=0;i<m;i++)
    for(int j=0;j<n;j++){
      int k = j*m+i;
      std::vector<pair<int, int>> vec;
      vec.emplace_back(i+1, j);
      vec.emplace_back(i-1, j);
      vec.emplace_back(i, j+1);
      vec.emplace_back(i, j-1);
      for(auto iter: vec){
        int ii = iter.first, jj = iter.second;
        if (ii>=0 && ii<m && jj>=0 && jj<n){
          ivec.push_back(k);
          jvec.push_back(jj*m+ii);
          vvec.push_back(K[k]);
          ivec.push_back(k);
          jvec.push_back(k);
          vvec.push_back(-K[k]);
        }
        else{
          pair<int, int> ed = std::make_pair(-1,-1);
          if (ii<=-1){
            ed = std::make_pair(j*(m+1), (j+1)*(m+1));
          }
          else if(ii>=m){
            ed = std::make_pair(j*(m+1)+m, (j+1)*(m+1)+m);
          }
          else if(jj<=-1){
            ed = std::make_pair(i, i+1);
          }
          else if(jj>=n){
            ed = std::make_pair(n*(m+1)+i, n*(m+1)+i+1);
          }
          if (bval.count(ed)>0){
            ivec.push_back(k);
            jvec.push_back(k);
            vvec.push_back(-2*K[k]);
            rhs[k] += 2*K[k]*bval[ed];
          }
        }
      }
    }

    int nn = ivec.size();
    TensorShape ii_shape({nn});
    TensorShape jj_shape({nn});
    TensorShape vv_shape({nn});

    Tensor* ii = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, ii_shape, &ii));
    Tensor* jj = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, jj_shape, &jj));
    Tensor* vv = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, vv_shape, &vv));

    auto ii_tensor = ii->flat<int64>().data();
    auto jj_tensor = jj->flat<int64>().data();
    auto vv_tensor = vv->flat<double>().data();

    for(int i=0;i<nn;i++){
        ii_tensor[i] = ivec[i];
        jj_tensor[i] = jvec[i];
        vv_tensor[i] = vvec[i];
    }
}

void backward(
  double *grad_pval, 
  double *grad_K, 
  const double *grad_vv, const double *grad_rhs, const double *rhs, 
  const double *K, const int64 *bc, const double *pval, int N, int m, int n, double h){
  map<pair<int,int>, double> bval; 
  map<pair<int,int>, int> pval_map;
  for(int i=0;i<N;i++){
    int a = std::min(bc[2*i]-1, bc[2*i+1]-1);
    int b = std::max(bc[2*i]-1, bc[2*i+1]-1);
    bval[std::make_pair(a, b)] = pval[i];
    pval_map[std::make_pair(a, b)] = i;
  }
  // vector<int> ivec, jvec;
  // vector<double> vvec;
  int cnt = 0;
  for(int i=0;i<m;i++)
    for(int j=0;j<n;j++){
      int k = j*m+i;
      std::vector<pair<int, int>> vec;
      vec.emplace_back(i+1, j);
      vec.emplace_back(i-1, j);
      vec.emplace_back(i, j+1);
      vec.emplace_back(i, j-1);
      for(auto iter: vec){
        int ii = iter.first, jj = iter.second;
        if (ii>=0 && ii<m && jj>=0 && jj<n){
          // ivec.push_back(k);
          // jvec.push_back(jj*m+ii);
          // vvec.push_back(K[k]);
          // ivec.push_back(k);
          // jvec.push_back(k);
          // vvec.push_back(-K[k]);
          grad_K[k] += grad_vv[cnt++];
          grad_K[k] -= grad_vv[cnt++];
        }
        else{
          pair<int, int> ed = std::make_pair(-1,-1);
          if (ii<=-1){
            ed = std::make_pair(j*(m+1), (j+1)*(m+1));
          }
          else if(ii>=m){
            ed = std::make_pair(j*(m+1)+m, (j+1)*(m+1)+m);
          }
          else if(jj<=-1){
            ed = std::make_pair(i, i+1);
          }
          else if(jj>=n){
            ed = std::make_pair(n*(m+1)+i, n*(m+1)+i+1);
          }
          if (bval.count(ed)>0){
            // ivec.push_back(k);
            // jvec.push_back(k);
            // vvec.push_back(-2*K[k]);
            // rhs[k] = 2*K[k]*bval[ed];
            grad_K[k] -= 2*grad_vv[cnt++];
            grad_K[k] += grad_rhs[k] * 2 * bval[ed];
            grad_pval[pval_map[ed]] +=  grad_rhs[k]*2*K[k];
          }
        }
      }
    }
}