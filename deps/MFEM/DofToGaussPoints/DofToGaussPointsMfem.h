#include "../Common.h"

// u(x_1) u(x_2) u(x_3)
// u(x) = u(p1*x1+p2*x2+p3*x3) = p1 * u(x1) + p2 * u(x2) + p3 *u(x3)
namespace MFEM{
    void DofToGaussPointsMfem_forward(double *out, const double *u){
        int k = 0;
        for (int i = 0; i < mmesh.nelem; i++){
            auto elem = mmesh.elements[i];
            for (int j = 0; j<elem->ngauss; j++){
                out[k] = 0.0;
                for (int r = 0; r < elem->ndof; r++)
                    out[k] += u[elem->dof[r]] * elem->h(r, j);
                k++;
            }
        }
    }

    void DofToGaussPointsMfem_backward(
        double *grad_u,
        const double *grad_out,
        const double *out, const double *u){
        int k = 0;
        for (int i = 0; i < mmesh.nelem; i++){
            auto elem = mmesh.elements[i];
            for (int j = 0; j<elem->ngauss; j++){
                for (int r = 0; r < elem->ndof; r++)
                    grad_u[elem->dof[r]] += grad_out[k] * elem->h(r, j);
                k++;
            }
        }
    }
}

extern "C" void DofToGaussPointsMfem_forward_Julia(double *out, const double *u){
    MFEM::DofToGaussPointsMfem_forward(out, u);
}