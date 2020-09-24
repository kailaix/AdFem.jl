#include "../Common.h"

namespace MFEM{
    void ComputeFemStiffnessMatrixMfem_forward(
        int64 * indices, double * vv, 
        const double *hmat){
        int elem_ndof = mmesh.elem_ndof;
        Eigen::MatrixXd B(3, elem_ndof*2);
        Eigen::Matrix3d K;
        K.setZero();
        B.setZero();
        int k = 0;
        int k0 = 0;
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
                    
                
                Eigen::MatrixXd NN = B.transpose() * K * B * elem->w[j];
                Eigen::VectorXi dofs(2*elem_ndof);
                for(int p = 0; p < elem_ndof; p++){
                    dofs[p] = elem->dof[p];
                    dofs[p+elem_ndof] = elem->dof[p] + mmesh.ndof;
                }

                for(int l = 0; l < elem_ndof * 2; l++){
                    for(int s = 0; s < elem_ndof * 2; s ++){
                        int idx1 = dofs[l];
                        int idx2 = dofs[s];
                        indices[2*k] = idx1;
                        indices[2*k+1] = idx2;
                        vv[k] = NN(l, s);
                        k ++; 
                    }
                }
            }
        }
    }

    void ComputeFemStiffnessMatrixMfem_backward(
        double * grad_hmat,
        const double * grad_vv){
        int elem_ndof = mmesh.elem_ndof;
        Eigen::MatrixXd B(3, elem_ndof*2);
        Eigen::MatrixXd K(2*elem_ndof, 2*elem_ndof);
        int k0 = 0;
        int k = 0;
        B.setZero();
        for(int i = 0; i<mmesh.nelem; i++){
            NNFEM_Element * elem = mmesh.elements[i];
            for (int j = 0; j< elem->ngauss; j++){
                for (int r = 0; r < elem_ndof; r ++){
                    B(0, r) = elem->hx(r, j);
                    B(1, r + elem_ndof) = elem->hy(r, j);
                    B(2, r) = elem->hy(r, j);
                    B(2, r + elem_ndof) = elem->hx(r, j);
                }


                for(int l = 0; l < elem_ndof * 2; l++){
                    for(int s = 0; s < elem_ndof * 2; s ++){
                        K(l, s) = grad_vv[k++];
                    }
                }
                Eigen::MatrixXd NN =  B * K * B.transpose() * elem->w[j];                
                for(int p = 0; p < 3; p++)
                    for (int q = 0; q < 3; q++)
                        grad_hmat[k0++] = NN(p, q);
            }
        }
    }
}

extern "C" void ComputeFemStiffnessMatrixMfem_forward_Julia(
        int64 * indices, double * vv, 
        const double *hmat){
    MFEM::ComputeFemStiffnessMatrixMfem_forward(indices, vv, hmat);
}