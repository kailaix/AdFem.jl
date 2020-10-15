#include "../Common.h"

namespace MFEM{
    void EvalStrainOnGaussPts_forward(double *epsilon, const double *u){
        int elem_ndof = mmesh.elem_ndof;
        for(int i = 0; i<mmesh.nelem; i++){
            NNFEM_Element * elem = mmesh.elements[i];
            for (int j = 0; j< elem->ngauss; j++){
                for (int p = 0; p < elem_ndof; p++){
                    epsilon[(i*elem->ngauss+j)*3] += elem->hx(p, j) * u[elem->dof[p]];
                    epsilon[(i*elem->ngauss+j)*3+1] += elem->hy(p, j) * u[elem->dof[p] + mmesh.ndof];
                    epsilon[(i*elem->ngauss+j)*3+2] += elem->hy(p, j) * u[elem->dof[p]] + elem->hx(p, j) * u[elem->dof[p] + mmesh.ndof];
                }
            }
        }
    }

    void EvalStrainOnGaussPts_backward(double *grad_u, const double *grad_epsilon){
        int elem_ndof = mmesh.elem_ndof;
        for(int i = 0; i<mmesh.nelem; i++){
            NNFEM_Element * elem = mmesh.elements[i];
            for (int j = 0; j< elem->ngauss; j++){
                for (int p = 0; p < elem_ndof; p++){
                    grad_u[elem->dof[p]] += elem->hx(p, j) * grad_epsilon[(i*elem->ngauss+j)*3] + elem->hy(p, j) * grad_epsilon[(i*elem->ngauss+j)*3+2];
                    grad_u[elem->dof[p] + mmesh.ndof] += elem->hy(p, j) * grad_epsilon[(i*elem->ngauss+j)*3+1] + elem->hx(p, j) * grad_epsilon[(i*elem->ngauss+j)*3+2];
                }
            }
        }
    }
}

extern "C" void EvalStrainOnGaussPts_forward_Julia(double *epsilon, const double *u){
    MFEM::EvalStrainOnGaussPts_forward(epsilon, u);
}