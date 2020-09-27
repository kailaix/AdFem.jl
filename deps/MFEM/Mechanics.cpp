#include "Common.h"

extern "C" void mfem_compute_von_mises_stress_term(double *sigma, const double *hmat, const double *u){
    int elem_ndof = mmesh.elem_ndof;
    Eigen::MatrixXd B(3, elem_ndof*2);
    Eigen::Matrix3d K;
    K.setZero();
    B.setZero();
    int k = 0;
    int k0 = 0;
    int s = 0;
    for(int i = 0; i<mmesh.nelem; i++){
        NNFEM_Element * elem = mmesh.elements[i];
        for (int j = 0; j< elem->ngauss; j++){
            for (int r = 0; r < elem_ndof; r ++){
                B(0, r) = elem->hx(r, j);
                B(1, r + elem_ndof) = elem->hy(r, j);
                B(2, r) = elem->hy(r, j);
                B(2, r + elem_ndof) = elem->hx(r, j);
            }
            for (int p = 0; p < 3; p++)
                for(int q = 0; q <3; q++)
                    K(p, q) = hmat[k0++];
                
            
            Eigen::MatrixXd NN = K * B;
            Eigen::VectorXd ulocal(2*elem_ndof);
            for(int p = 0; p < elem_ndof; p++){
                ulocal[p] = u[elem->dof[p]];
                ulocal[p+elem_ndof] = u[elem->dof[p] + mmesh.ndof];
            }
            Eigen::VectorXd Sigma = NN * ulocal;
            double sigma11 = Sigma[0], sigma22 = Sigma[1], sigma12 = Sigma[2];
            sigma[s++] = sqrt(sigma11 * sigma11 - sigma11 * sigma22 + sigma22 * sigma22 + 3 * sigma12 * sigma12);
        }
    }
}