#include "../Common.h"
namespace MFEM{
  void FemLaplaceScalarT_forward(int64 *indices, double *vv, const double *kappa){
    int s = 0;
    int nz = 0;
    int elem_ndof = mmesh3.elem_ndof;
    for(int i = 0; i<mmesh3.nelem; i++){
        NNFEM_Element3 * elem = mmesh3.elements[i];
        Eigen::MatrixXd D(elem_ndof, 3);
        for (int k = 0; k<elem->ngauss; k++){
          for (int r = 0; r < elem_ndof; r++){
            D(r, 0) = elem->hx(r, k);
            D(r, 1) = elem->hy(r, k);
            D(r, 2) = elem->hz(r, k);
          }
          Eigen::MatrixXd N = D * D.transpose() * kappa[s++] * elem->w[k];
          for (int p = 0; p < elem_ndof; p ++)
            for (int q = 0; q< elem_ndof; q++){
              indices[2*nz] = elem->dof[p];
              indices[2*nz+1] = elem->dof[q];
              vv[nz] = N(p, q);
              nz ++;
            }
        }
        
    }
  }

  void FemLaplaceScalarT_backward(
    double *grad_kappa,
    const double * grad_vv,
    const int64 *indices, const double *vv, const double *kappa){
    int nz = 0;
    int s = 0;
    int elem_ndof = mmesh3.elem_ndof;
    for(int i = 0; i<mmesh3.nelem; i++){
        NNFEM_Element3 * elem = mmesh3.elements[i];
        Eigen::MatrixXd D(elem_ndof, 3);
        for (int k = 0; k<elem->ngauss; k++){
          for (int r = 0; r < elem_ndof; r++){
            D(r, 0) = elem->hx(r, k);
            D(r, 1) = elem->hy(r, k);
            D(r, 2) = elem->hz(r, k);
          }
          Eigen::MatrixXd N = D * D.transpose() * elem->w[k];
          double v = 0.0;
          for (int p = 0; p < elem_ndof; p ++)
            for (int q = 0; q< elem_ndof; q++){
              v += grad_vv[nz] * N(p, q);
              nz ++;
            }
          grad_kappa[s++] = v;
        }
        
    }
  }

}


extern "C" void FemLaplaceScalarT_forward_Julia(int64 *indices, double *vv, const double *kappa){
  MFEM::FemLaplaceScalarT_forward(indices, vv, kappa);
}