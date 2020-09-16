#include "../Common.h"

// u(x) u(x2) u(x3)
// u(x) = u(p1*x1+p2*x2+p3*x3) = p1 * u(x1) + p2 * u(x2) + p3 *u(x3)
// du(x) = du(p1*x1+p2*x2+p3*x3) = p1 * du(x1) + p2 * du(x2) + p3 *du(x3)

namespace MFEM{
    void FemGradMfem_forward(double *out, const double *u){
        int k = 0;
        for (int i = 0; i < mmesh.nelem; i++){
            auto elem = mmesh.elements[i];
            for (int j = 0; j<elem->ngauss; j++){
                out[k] = u[elem->node[0]] * elem->hx(0, j) + 
                           u[elem->node[1]] * elem->hx(1, j) + 
                            u[elem->node[2]] * elem->hx(2, j);
                k++;
                out[k] = u[elem->node[0]] * elem->hy(0, j) + 
                           u[elem->node[1]] * elem->hy(1, j) + 
                            u[elem->node[2]] * elem->hy(2, j);
                k++;
            }
        }
    }

    void FemGradMfem_backward(
        double *grad_u,
        const double *grad_out,
        const double *out, const double *u){
        int k = 0;
        for (int i = 0; i < mmesh.nelem; i++){
            auto elem = mmesh.elements[i];
            for (int j = 0; j<elem->ngauss; j++){
                grad_u[elem->node[0]] += grad_out[k] * elem->hx(0, j);
                grad_u[elem->node[1]] += grad_out[k] * elem->hx(1, j);
                grad_u[elem->node[2]] += grad_out[k] * elem->hx(2, j);
                k++;
                grad_u[elem->node[0]] += grad_out[k] * elem->hy(0, j);
                grad_u[elem->node[1]] += grad_out[k] * elem->hy(1, j);
                grad_u[elem->node[2]] += grad_out[k] * elem->hy(2, j);
                k++;
            }
        }
    }
}