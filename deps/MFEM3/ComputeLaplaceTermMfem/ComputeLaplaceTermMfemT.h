#include "../Common.h"

namespace MFEM{
    void ComputeLaplaceTermMfemT_forward(double *out, const double *nu,  const double *u){
        int elem_ndof = mmesh3.elem_ndof;
        for(int i = 0; i<mmesh3.nelem; i++){
            NNFEM_Element3 * elem = mmesh3.elements[i];
            for (int j = 0; j< elem->ngauss; j++){
                double nu_val = nu[i*elem->ngauss + j];
                for (int p = 0; p < elem_ndof; p++){
                    for (int q = 0; q < elem_ndof; q++){
                            out[elem->dof[p]] += nu_val * (elem->hx(p, j) * elem->hx(q, j) * u[elem->dof[q]] * elem->w[j] + 
                                                        elem->hy(p, j) * elem->hy(q, j) * u[elem->dof[q]] * elem->w[j] +
                                                            elem->hz(p, j) * elem->hz(q, j) * u[elem->dof[q]] * elem->w[j]); 
                            // printf("hz = %f %f %f\n", elem->hx(p, j), elem->hy(p, j), elem->hz(p, j));
                    }
                        
                }
            }
        }
    }

    void ComputeLaplaceTermMfemT_backward(
        double *grad_nu, double *grad_u,
         const double *grad_out, 
        const double *out, const double *nu, const double *u){
        int elem_ndof = mmesh3.elem_ndof;
        for(int i = 0; i<mmesh3.nelem; i++){
            NNFEM_Element3 * elem = mmesh3.elements[i];
            for (int j = 0; j< elem->ngauss; j++){
                double nu_val = nu[i*elem->ngauss + j];
                for (int p = 0; p < elem_ndof; p++){
                    for (int q = 0; q < elem_ndof; q++){
                        grad_nu[i*elem->ngauss + j] += grad_out[elem->dof[p]] * 
                                                    (elem->hx(p, j) * elem->hx(q, j) * u[elem->dof[q]] * elem->w[j] + 
                                                        elem->hy(p, j) * elem->hy(q, j) * u[elem->dof[q]] * elem->w[j] +
                                                            elem->hz(p, j) * elem->hz(q, j) * u[elem->dof[q]] * elem->w[j]);
                        grad_u[elem->dof[q]] += grad_out[elem->dof[p]] * nu_val * elem->hx(p, j) * elem->hx(q, j) * elem->w[j];
                        grad_u[elem->dof[q]] += grad_out[elem->dof[p]] * nu_val * elem->hy(p, j) * elem->hy(q, j) * elem->w[j];
                        grad_u[elem->dof[q]] += grad_out[elem->dof[p]] * nu_val * elem->hz(p, j) * elem->hz(q, j) * elem->w[j];
                    } 
                }
            }
        }
    }
}

extern "C" void ComputeLaplaceTermMfem3_forward_Julia(double *out, const double *nu,  const double *u){
    MFEM::ComputeLaplaceTermMfemT_forward(out, nu, u);
}