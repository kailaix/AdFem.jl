#include "../Common.h"
namespace MFEM{
  // integrate_T( f, phi ) = f(x_1) * phi(x_1) * w_1 + f(x_2) * phi(x_2) * w_2 + ...
  void FemSourceScalar_forward(double * rhs, const double *f){
    int k = 0;
    int elem_ndof = mmesh.elem_ndof;
    for (int i = 0; i < mmesh.nelem; i++){
      auto elem = mmesh.elements[i];
      for (int j = 0; j<elem->ngauss; j++){
        for(int r = 0; r<elem_ndof; r++)
            rhs[elem->dof[r]] += f[k] * elem->h(r, j) * elem->w[j];
        k ++;
      }
    }
  }

  void FemSourceScalar_backward(
    double *grad_f,
    const double *grad_rhs, 
    const double *rhs, const double *f){
    int k = 0;
    int elem_ndof = mmesh.elem_ndof;
    for (int i = 0; i < mmesh.nelem; i++){
      auto elem = mmesh.elements[i];
      for (int j = 0; j<elem->ngauss; j++){
        grad_f[k] = 0.0;
        for(int r = 0; r<elem_ndof; r++)
          grad_f[k] += elem->h(r, j) * elem->w[j] * grad_rhs[elem->dof[r]];
        k ++;
      }
    }
  }
}

extern "C" void FemSourceScalar_forward_Julia(double * rhs, const double *f){
  MFEM::FemSourceScalar_forward(rhs, f);
}