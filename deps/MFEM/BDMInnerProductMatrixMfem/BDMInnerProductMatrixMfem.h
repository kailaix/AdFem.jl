#include "../Common.h"

namespace MFEM{
  // alpha * sigma + beta * tr(sigma) * I 
  void BDMInnerProductMatrixMfem_forward(int64 *indices, double *vv, 
            const double *alpha, const double *beta){
    int elem_ndof = mmesh.elem_ndof;    
    int k = 0, k0 = -1;
    for(int i = 0; i<mmesh.nelem; i++){
        NNFEM_Element * elem = mmesh.elements[i];
        for (int j = 0; j< elem->ngauss; j++){
            k0++;
            // printf("Processing %d/%d\n", i, j);
            for (int r = 0; r < elem_ndof; r++){
                for (int s = 0; s < elem_ndof; s++){
                  // alpha * < sigma, tau > 
                  indices[2*k] = elem->dof[s];
                  indices[2*k+1] = elem->dof[r];
                  vv[k] = alpha[k0] * (elem->BDMx(r, j) * elem->BDMx(s, j) +
                                      elem->BDMy(r, j) * elem->BDMy(s, j)) * elem->w[j];
                  // printf("Values: vv[%d] = %f\n", k, vv[k]);
                  k++;

                  indices[2*k] = elem->dof[s] + mmesh.ndof;
                  indices[2*k+1] = elem->dof[r] + mmesh.ndof;
                  vv[k] = vv[k-1];
                  k++;

                  // beta * tr(sigma) * tr(tau)
                  indices[2*k] = elem->dof[s];
                  indices[2*k+1] = elem->dof[r];
                  vv[k] = beta[k0] * (elem->BDMx(r, j) * elem->BDMx(s, j)) * elem->w[j];
                  // printf("Values: vv[%d] = %f\n", k, vv[k]);
                  k++;

                  indices[2*k] = elem->dof[s] + mmesh.ndof;
                  indices[2*k+1] = elem->dof[r] + mmesh.ndof;
                  vv[k] = beta[k0] * (elem->BDMy(r, j) * elem->BDMy(s, j)) * elem->w[j];
                  // printf("Values: vv[%d] = %f\n", k, vv[k]);
                  k++;


                  indices[2*k] = elem->dof[s] + mmesh.ndof;
                  indices[2*k+1] = elem->dof[r];
                  vv[k] = beta[k0] * (elem->BDMx(r, j) * elem->BDMy(s, j)) * elem->w[j];
                  // printf("Values: vv[%d] = %f\n", k, vv[k]);
                  k++;

                  indices[2*k] = elem->dof[s] ;
                  indices[2*k+1] = elem->dof[r] + mmesh.ndof;
                  vv[k] = beta[k0] * (elem->BDMy(r, j) * elem->BDMx(s, j)) * elem->w[j];
                  // printf("Values: vv[%d] = %f\n", k, vv[k]);
                  k++;
                  
                
                }
            }
        }
    }
    // printf("ccall finished\n");
  }

  // alpha * sigma
  void BDMInnerProductMatrixMfem1_forward(int64 *indices, double *vv, 
            const double *alpha){
    int elem_ndof = mmesh.elem_ndof;    
    int k = 0, k0 = -1;
    for(int i = 0; i<mmesh.nelem; i++){
        NNFEM_Element * elem = mmesh.elements[i];
        for (int j = 0; j< elem->ngauss; j++){
            k0++;
            // printf("Processing %d/%d\n", i, j);
            for (int r = 0; r < elem_ndof; r++){
                for (int s = 0; s < elem_ndof; s++){
                  // alpha * < sigma, tau > 
                  indices[2*k] = elem->dof[s];
                  indices[2*k+1] = elem->dof[r];
                  vv[k] = alpha[k0] * (elem->BDMx(r, j) * elem->BDMx(s, j) +
                                      elem->BDMy(r, j) * elem->BDMy(s, j)) * elem->w[j];
                  // printf("Values: vv[%d] = %f\n", k, vv[k]);
                  k++;                
                }
            }
        }
    }
}


  void BDMInnerProductMatrixMfem_backward(){
    
  }
}

extern "C" void BDMInnerProductMatrixMfem_forward_Julia(int64 *indices, double *vv, 
            const double *alpha, const double *beta){
      MFEM::BDMInnerProductMatrixMfem_forward(indices, vv, alpha, beta);
}

extern "C" void BDMInnerProductMatrixMfem1_forward_Julia(int64 *indices, double *vv, 
            const double *alpha){
      MFEM::BDMInnerProductMatrixMfem1_forward(indices, vv, alpha);
}