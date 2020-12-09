#include "../Common.h"
namespace MFEM{
  // integrate_T( f, phi ) = f(x_1) * phi(x_1) * w_1 + f(x_2) * phi(x_2) * w_2 + ...
  void FemSourceScalarT_forward(double * rhs, const double *f){
    int k = 0;
    int elem_ndof = mmesh3.elem_ndof;
    for (int i = 0; i < mmesh3.nelem; i++){
      auto elem = mmesh3.elements[i];
      for (int j = 0; j<elem->ngauss; j++){
        for(int r = 0; r<elem_ndof; r++)
            rhs[elem->dof[r]] += f[k] * elem->h(r, j) * elem->w[j];
        k ++;
      }
    }
  }

  void FemSourceScalarT_backward(
    double *grad_f,
    const double *grad_rhs, 
    const double *rhs, const double *f){
    int k = 0;
    int elem_ndof = mmesh3.elem_ndof;
    for (int i = 0; i < mmesh3.nelem; i++){
      auto elem = mmesh3.elements[i];
      for (int j = 0; j<elem->ngauss; j++){
        grad_f[k] = 0.0;
        for(int r = 0; r<elem_ndof; r++)
          grad_f[k] += elem->h(r, j) * elem->w[j] * grad_rhs[elem->dof[r]];
        k ++;
      }
    }
  }
}

extern "C" void FemSourceScalarT_forward_Julia(double * rhs, const double *f){
  MFEM::FemSourceScalarT_forward(rhs, f);
}