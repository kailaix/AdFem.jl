#include "../Common.h"

namespace MFEM{
    void ComputeFemMassMatrix1_forward(int64 *indices, double *vv, const double *rho){
        Eigen::MatrixXd N(3, 1);
        int k = 0;
        int k0 = 0;
        for(int i = 0; i<mmesh.nelem; i++){
            NNFEM_Element * elem = mmesh.elements[i];
            for (int j = 0; j< elem->ngauss; j++){
                N(0, 0) = elem->h(0, j);
                N(1, 0) = elem->h(1, j);
                N(2, 0) = elem->h(2, j);
                Eigen::MatrixXd NN = N * N.transpose() * rho[k0++] * elem->w[j];

                for(int l = 0; l < 3; l++){
                    for(int s = 0; s < 3; s ++){
                        int idx1 = elem->node[l];
                        int idx2 = elem->node[s];
                        indices[2*k] = idx1;
                        indices[2*k+1] = idx2;
                        vv[k] = NN(l, s);
                        k ++; 
                    }
                }
            }
        }
    }

    void ComputeFemMassMatrix1_backward(
        double * grad_rho, 
        const double *grad_vv, 
        const double *vv, const double *rho){
        Eigen::MatrixXd N(3, 1);
        int k = 0, k0 = 0;
        for(int i = 0; i<mmesh.nelem; i++){
            NNFEM_Element * elem = mmesh.elements[i];
            for (int j = 0; j< elem->ngauss; j++){
                N(0, 0) = elem->h(0, j);
                N(1, 0) = elem->h(1, j);
                N(2, 0) = elem->h(2, j);
                Eigen::MatrixXd NN = N * N.transpose() * elem->w[j];
                for(int l = 0; l < 3; l++){
                    for(int s = 0; s < 3; s ++){
                        int idx1 = elem->node[l];
                        int idx2 = elem->node[s];
                        grad_rho[k0] += NN(l, s) * grad_vv[k];
                        k ++; 
                    }
                }
                k0++;
            }
        }
    }
}