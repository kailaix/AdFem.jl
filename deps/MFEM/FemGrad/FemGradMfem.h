#include "../Common.h"

// u(x) u(x2) u(x3)
// u(x) = u(p1*x1+p2*x2+p3*x3) = p1 * u(x1) + p2 * u(x2) + p3 *u(x3)
// du(x) = du(p1*x1+p2*x2+p3*x3) = p1 * du(x1) + p2 * du(x2) + p3 *du(x3)

namespace MFEM{
    void FemGradMfem_forward(double *out, const double *u){
        int k = 0;
        int elem_ndof = mmesh.elem_ndof;
        for (int i = 0; i < mmesh.nelem; i++){
            auto elem = mmesh.elements[i];
            for (int j = 0; j<elem->ngauss; j++){
                out[k] = 0.0;
                for (int r = 0; r < elem_ndof; r++) out[k] += u[elem->dof[r]] * elem->hx(r, j);
                k++;
                out[k] = 0.0;
                for (int r = 0; r < elem_ndof; r++) out[k] += u[elem->dof[r]] * elem->hy(r, j);
                k++;
            }
        }
    }

    void FemGradMfem_backward(
        double *grad_u,
        const double *grad_out,
        const double *out, const double *u){
        int elem_ndof = mmesh.elem_ndof;
        int k = 0;
        for (int i = 0; i < mmesh.nelem; i++){
            auto elem = mmesh.elements[i];
            for (int j = 0; j<elem->ngauss; j++){
                for (int r = 0; r < elem_ndof; r++)
                    grad_u[elem->dof[r]] += grad_out[k] * elem->hx(r, j);
                k++;
                for (int r = 0; r < elem_ndof; r++)
                    grad_u[elem->dof[r]] += grad_out[k] * elem->hy(r, j);
                k++;
            }
        }
    }
}