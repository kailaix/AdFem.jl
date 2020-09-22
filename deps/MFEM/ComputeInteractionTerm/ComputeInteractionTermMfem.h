#include "../Common.h"

namespace MFEM{
    void ComputeInteractionTermMfem_forward(double *out, const double *p){
        int elem_ndof = mmesh.elem_ndof;
        for(int i = 0; i<mmesh.nelem; i++){
            NNFEM_Element * elem = mmesh.elements[i];
            for (int j = 0; j< elem->ngauss; j++)
                for (int k = 0; k < elem_ndof; k++){
                    out[elem->dof[k]] += elem->hx(k, j) * elem->w[j] * p[i];
                    out[elem->dof[k] + mmesh.ndof] += elem->hy(k, j) * elem->w[j] * p[i];
                }
        }
    }

    void ComputeInteractionTermMfem_backward(
        double *grad_p, const double *grad_out, 
        const double *out, const double *p){
        int elem_ndof = mmesh.elem_ndof;
        for(int i = 0; i<mmesh.nelem; i++){
            NNFEM_Element * elem = mmesh.elements[i];
            for (int j = 0; j< elem->ngauss; j++)
                for (int k = 0; k < elem_ndof; k++){
                    grad_p[i] += elem->hx(k, j) * elem->w[j] * grad_out[elem->dof[k]] + 
                                        elem->hy(k, j) * elem->w[j] * grad_out[elem->dof[k] + mmesh.ndof];
                }
        }
    }
}