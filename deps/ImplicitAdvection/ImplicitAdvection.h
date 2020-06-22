#include <map>
#include <utility>
#include <algorithm>
#include <vector>

class IAForward{
public:
  std::vector<int64> ii_, jj_;
  std::vector<double> vv_;
  void forward(double *rhs,
    const int64 *bc, const double *bcval, int nbc, 
    const double *u, const double *v, int m, int n, double h);
  void copy_data(int64 *ii, int64*jj, double*vv);
};

void IAForward::copy_data(int64 *ii, int64*jj, double*vv){
  for(int i = 0; i < ii_.size(); i++){
    ii[i] = ii_[i];
    jj[i] = jj_[i];
    vv[i] = vv_[i];
  }
}

void IAForward::forward(
  double *rhs, 
  const int64 *bc, const double *bcval, int nbc, 
  const double *u, const double *v, int m, int n, double h){
  std::map<std::pair<int64, int64>, int> bcmap;
  for(int i=0;i<nbc;i++){
    bcmap[std::make_pair(bc[2*i], bc[2*i+1])] = bcval[i];
  }

  // V = h * u * (T_right - T_left) + h * v * (T_down - T_up)
  for(int i = 0;i<m; i++){
    for( int j = 0; j< n; j++){
      int k = j*m+i;

// T_down
      if(j==n-1){
        auto edge = std::make_pair((j+1)*m+i, (j+1)*m+i+1);
        if (bcmap.count(edge)){
          rhs[k] += h * v[k] * bcval[bcmap[edge]];
        }
      }else{
          ii_.push_back(k);
          jj_.push_back((j+1)*m+i);
          vv_.push_back(h*v[k]/2);

          ii_.push_back(k);
          jj_.push_back(k);
          vv_.push_back(h*v[k]/2);
      }
      

// T_up
      if(j==0){
        auto edge = std::make_pair(i, i+1);
        if (bcmap.count(edge)){
          rhs[k] -= h * v[k] * bcval[bcmap[edge]];
        }
      }else{
          ii_.push_back(k);
          jj_.push_back((j-1)*m+i);
          vv_.push_back(-h*v[k]/2);

          ii_.push_back(k);
          jj_.push_back(k);
          vv_.push_back(-h*v[k]/2);
      }


// T_right
      if(i==m-1){
        auto edge = std::make_pair(j*m+i+1, (j+1)*m+i+1);
        if (bcmap.count(edge)){
          rhs[k] += h * u[k] * bcval[bcmap[edge]];
        }
      }else{
          ii_.push_back(k);
          jj_.push_back(j*m+i+1);
          vv_.push_back(h*u[k]/2);

          ii_.push_back(k);
          jj_.push_back(k);
          vv_.push_back(h*u[k]/2);
      }

// T_left 
      if(i==0){
        auto edge = std::make_pair(j*m+i, (j+1)*m+i);
        if (bcmap.count(edge)){
          rhs[k] -= h * u[k] * bcval[bcmap[edge]];
        }
      }else{
          ii_.push_back(k);
          jj_.push_back(j*m+i-1);
          vv_.push_back(-h*u[k]/2);

          ii_.push_back(k);
          jj_.push_back(k);
          vv_.push_back(-h*u[k]/2);
      }

    }
  }

}

void IA_backward(
  double * grad_bc,  double *grad_u, double * grad_v, 
  const double * grad_vv, 
  const double *grad_rhs, 
  const int64 *bc, const double *bcval, int nbc, 
  const double *u, const double *v, int m, int n, double h){
  int idx = 0;
  std::map<std::pair<int64, int64>, int> bcmap;
  for(int i=0;i<nbc;i++){
    bcmap[std::make_pair(bc[2*i], bc[2*i+1])] = bcval[i];
  }

  // V = h * u * (T_right - T_left) + h * v * (T_down - T_up)
  for(int i = 0;i<m; i++){
    for( int j = 0; j< n; j++){
      int k = j*m+i;

// T_down
      if(j==n-1){
        auto edge = std::make_pair((j+1)*m+i, (j+1)*m+i+1);
        if (bcmap.count(edge)){
          // rhs[k] += h * v[k] * bcval[bcmap[edge]];
          grad_v[k] += grad_rhs[k] * h * bcval[bcmap[edge]];
          grad_bc[bcmap[edge]] += grad_rhs[k] * h * v[k];
        }
      }else{
          // ii_.push_back(k);
          // jj_.push_back((j+1)*m+i);
          // vv_.push_back(h*v[k]/2);

          // ii_.push_back(k);
          // jj_.push_back(k);
          // vv_.push_back(h*v[k]/2);
          grad_v[k] += grad_vv[idx++] * h/2;
          grad_v[k] += grad_vv[idx++] * h/2;
      }
      

// T_up
      if(j==0){
        auto edge = std::make_pair(i, i+1);
        if (bcmap.count(edge)){
          // rhs[k] -= h * v[k] * bcval[bcmap[edge]];
          grad_v[k] -= grad_rhs[k] * h * bcval[bcmap[edge]];
          grad_bc[bcmap[edge]] -= grad_rhs[k] * h * v[k];
        }
      }else{
          // ii_.push_back(k);
          // jj_.push_back((j-1)*m+i);
          // vv_.push_back(-h*v[k]/2);

          // ii_.push_back(k);
          // jj_.push_back(k);
          // vv_.push_back(-h*v[k]/2);
          grad_v[k] -= grad_vv[idx++] * h/2;
          grad_v[k] -= grad_vv[idx++] * h/2;
      }


// T_right
      if(i==m-1){
        auto edge = std::make_pair(j*m+i+1, (j+1)*m+i+1);
        if (bcmap.count(edge)){
          // rhs[k] += h * u[k] * bcval[bcmap[edge]];
          grad_u[k] += grad_rhs[k] * h * bcval[bcmap[edge]];
          grad_bc[bcmap[edge]] += grad_rhs[k] * h * u[k];
        }
      }else{
          // ii_.push_back(k);
          // jj_.push_back(j*m+i+1);
          // vv_.push_back(h*u[k]/2);

          // ii_.push_back(k);
          // jj_.push_back(k);
          // vv_.push_back(h*u[k]/2);
          grad_u[k] += grad_vv[idx++] * h/2;
          grad_u[k] += grad_vv[idx++] * h/2;
      }

// T_left 
      if(i==0){
        auto edge = std::make_pair(j*m+i, (j+1)*m+i);
        if (bcmap.count(edge)){
          // rhs[k] -= h * u[k] * bcval[bcmap[edge]];
          grad_u[k] -= grad_rhs[k] * h * bcval[bcmap[edge]];
          grad_bc[bcmap[edge]] -= grad_rhs[k] * h * u[k];
        }
      }else{
          // ii_.push_back(k);
          // jj_.push_back(j*m+i-1);
          // vv_.push_back(-h*u[k]/2);

          // ii_.push_back(k);
          // jj_.push_back(k);
          // vv_.push_back(-h*u[k]/2);
          grad_u[k] -= grad_vv[idx++] * h/2;
          grad_u[k] -= grad_vv[idx++] * h/2;
      }

    }
  }
}