#include "../Common.h"

namespace MFEM{
    void ComputeStrainEnergyTermMfem_forward(double *out, const double *sigma){
        int elem_ndof = mmesh.elem_ndof;
        for(int i = 0; i<mmesh.nelem; i++){
            NNFEM_Element * elem = mmesh.elements[i];
            for (int j = 0; j< elem->ngauss; j++){
                double s11 = sigma[3*(i*elem->ngauss + j)], s22 = sigma[3*(i*elem->ngauss + j)+1], s12 = sigma[3*(i*elem->ngauss + j)+2];
                for (int p = 0; p < elem_ndof; p++){
                        out[elem->dof[p]] += s11 * elem->hx(p, j) * elem->w[j]; 
                        out[elem->dof[p] + mmesh.ndof] += s22 * elem->hy(p, j) * elem->w[j];
                        out[elem->dof[p]] += s12 * elem->hy(p, j) * elem->w[j];       
                        out[elem->dof[p] + mmesh.ndof] += s12 * elem->hx(p, j) * elem->w[j];       
                }
            }
        }
    }

    void ComputeStrainEnergyTermMfem_backward(
        double *grad_sigma, 
        const double *grad_out){
        int elem_ndof = mmesh.elem_ndof;
        for(int i = 0; i<mmesh.nelem; i++){
            NNFEM_Element * elem = mmesh.elements[i];
            for (int j = 0; j< elem->ngauss; j++){
                for (int p = 0; p < elem_ndof; p++){
                    grad_sigma[3*(i*elem->ngauss + j)] += elem->hx(p, j) * elem->w[j] * grad_out[elem->dof[p]];
                    grad_sigma[3*(i*elem->ngauss + j)+1] += elem->hy(p, j) * elem->w[j] * grad_out[elem->dof[p] + mmesh.ndof];
                    grad_sigma[3*(i*elem->ngauss + j)+2] += elem->hy(p, j) * elem->w[j] * grad_out[elem->dof[p]] + elem->hx(p, j) * elem->w[j] * grad_out[elem->dof[p] + mmesh.ndof];
                }
            }
        }
    }
}

extern "C" void ComputeStrainEnergyTermMfem_forward_Julia(double *out, const double *sigma){
    MFEM::ComputeStrainEnergyTermMfem_forward(out, sigma);
}