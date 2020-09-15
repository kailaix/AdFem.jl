#include "../Common.h"
namespace MFEM{
  void FemSourceScalar_forward(double * rhs, const double *f){
    int k = 0;
    for (int i = 0; i < mmesh.nelem; i++){
      auto elem = mmesh.elements[i];
      for (int j = 0; j<elem->ngauss; j++){
        rhs[elem->node[0]] += f[k] * elem->h(0, j) * elem->w[j];
        rhs[elem->node[1]] += f[k] * elem->h(1, j) * elem->w[j];
        rhs[elem->node[2]] += f[k] * elem->h(2, j) * elem->w[j];
        k ++;
      }
    }
  }

  void FemSourceScalar_backward(
    double *grad_f,
    const double *grad_rhs, 
    const double *rhs, const double *f){
    int k = 0;
    for (int i = 0; i < mmesh.nelem; i++){
      auto elem = mmesh.elements[i];
      for (int j = 0; j<elem->ngauss; j++){
        grad_f[k] = elem->h(0, j) * elem->w[j] * grad_rhs[elem->node[0]] + 
                    elem->h(1, j) * elem->w[j] * grad_rhs[elem->node[1]] + 
                    elem->h(2, j) * elem->w[j] * grad_rhs[elem->node[2]];
        k ++;
      }
    }
  }
}