#include "../Common.h"

namespace MFEM{
    void ComputeFemAdvectionMatrix_forward(int64 *indices, double *vv, 
        const double *u, const double *v){
        int elem_ndof = mmesh.elem_ndof;
        Eigen::MatrixXd N(elem_ndof, 1), Nx(1, elem_ndof), Ny(1, elem_ndof);
        int k = 0;
        int k0 = 0;
        for(int i = 0; i<mmesh.nelem; i++){
            NNFEM_Element * elem = mmesh.elements[i];
            for (int j = 0; j< elem->ngauss; j++){
                for (int r = 0; r<elem_ndof; r++){
                    N(r, 0) = elem->h(r, j);
                    Nx(0, r) = elem->hx(r, j);
                    Ny(0, r) = elem->hy(r, j);
                }
                
                Eigen::MatrixXd NN = N * (Nx * u[k0] + Ny * v[k0]) * elem->w[j];
                k0++;

                for(int l = 0; l < elem_ndof; l++){
                    for(int s = 0; s < elem_ndof; s ++){
                        int idx1 = elem->dof[l];
                        int idx2 = elem->dof[s];
                        indices[2*k] = idx1;
                        indices[2*k+1] = idx2;
                        vv[k] = NN(l, s);
                        k ++; 
                    }
                }
            }
        }
    }

    void ComputeFemAdvectionMatrix_backward(
        double *grad_u, double *grad_v, 
        const double *grad_vv){
        int elem_ndof = mmesh.elem_ndof;
        Eigen::MatrixXd N(elem_ndof, 1), Nx(1, elem_ndof), Ny(1, elem_ndof);
        int k = 0;
        int k0 = 0;
        for(int i = 0; i<mmesh.nelem; i++){
            NNFEM_Element * elem = mmesh.elements[i];
            for (int j = 0; j< elem->ngauss; j++){
                for (int r = 0; r<elem_ndof; r++){
                    N(r, 0) = elem->h(r, j);
                    Nx(0, r) = elem->hx(r, j);
                    Ny(0, r) = elem->hy(r, j);
                }
                
                Eigen::MatrixXd NX = N * Nx  * elem->w[j];
                Eigen::MatrixXd NY = N * Ny  * elem->w[j];
                

                for(int l = 0; l < elem_ndof; l++){
                    for(int s = 0; s < elem_ndof; s ++){
                        int idx1 = elem->dof[l];
                        int idx2 = elem->dof[s];
                        grad_u[k0] += NX(l, s) * grad_vv[k];
                        grad_v[k0] += NY(l, s) * grad_vv[k];
                        k ++; 
                    }
                }
                k0++;
            }
        }
    }
}