#include "../Common.h"

// u(x_1) u(x_2) u(x_3)
// u(x) = u(p1*x1+p2*x2+p3*x3) = p1 * u(x1) + p2 * u(x2) + p3 *u(x3)
namespace MFEM{
    void FemToGaussPointsMfem_forward(double *out, const double *u){
        int k = 0;
        for (int i = 0; i < mmesh.nelem; i++){
            auto elem = mmesh.elements[i];
            for (int j = 0; j<elem->ngauss; j++){
                out[k] = u[elem->node[0]] * elem->h(0, j) + 
                           u[elem->node[1]] * elem->h(1, j) + 
                            u[elem->node[2]] * elem->h(2, j);
                k++;
            }
        }
    }

    void FemToGaussPointsMfem_backward(
        double *grad_u,
        const double *grad_out,
        const double *out, const double *u){
        int k = 0;
        for (int i = 0; i < mmesh.nelem; i++){
            auto elem = mmesh.elements[i];
            for (int j = 0; j<elem->ngauss; j++){
                // grad_u[elem->node[0]] += grad_out[k] * elem->h(0, j);
                // grad_u[elem->node[1]] += grad_out[k] * elem->h(1, j);
                // grad_u[elem->node[2]] += grad_out[k] * elem->h(2, j);
                // k++;
            }
        }
    }
}