#include "../Common.h"
namespace MFEM{
    void fill_left_right_matrix(int lr, double n1, double n2, MatrixXd &MAT){
        switch (lr)
        {
        case -1:
            MAT << 1-n1*n1, -n1*n2, -n1*n2, 1-n2*n2;
            break;
        case 1:
            MAT << n1*n1, n1*n2, n1*n2, n2*n2;
            break;
        default:
            printf("ERROR: ComputePerpendicularParallelGradient: left/right must be -1 or 1.\n");
            break;
        }
    }
    void ComputePerpendicularParallelGradientForward(double *nmat, 
            const double *nv, const double *cmat, const int64 *left, const int64 *right){
        MatrixXd LEFT(2,2), RIGHT(2,2), C(2,2);
        int k = 0;
        for(int i = 0; i<mmesh.nelem; i++){
            NNFEM_Element * elem = mmesh.elements[i];
            for (int j = 0; j< elem->ngauss; j++){
                C << cmat[4*k], cmat[4*k+1], cmat[4*k+2], cmat[4*k+3];
                fill_left_right_matrix(*left, nv[2*k], nv[2*k+1], LEFT);
                fill_left_right_matrix(*right, nv[2*k], nv[2*k+1], RIGHT);
                auto N = LEFT*C*RIGHT;
                nmat[4*k] = N(0,0);
                nmat[4*k+1] = N(0,1);
                nmat[4*k+2] = N(1,0);
                nmat[4*k+3] = N(1,1);
                k++;
            }
        }
    }

    void ComputePerpendicularParallelGradientBackward(
            double *grad_cmat,
            const double *grad_nmat, const double *nmat, 
            const double *nv, const double *cmat, const int64 *left, const int64 *right){
        MatrixXd LEFT(2,2), RIGHT(2,2), C(2,2);
        int k = 0;
        for(int i = 0; i<mmesh.nelem; i++){
            NNFEM_Element * elem = mmesh.elements[i];
            for (int j = 0; j< elem->ngauss; j++){
                fill_left_right_matrix(*left, nv[2*k], nv[2*k+1], LEFT);
                fill_left_right_matrix(*right, nv[2*k], nv[2*k+1], RIGHT);
                for(int l = 0; l < 2; l++){
                    for (int m = 0; m < 2; m++){
                        int idx = 2*m+l;
                        grad_cmat[4*k+idx] = LEFT(0, l) * RIGHT(m, 0) * grad_nmat[4*k] + \
                                            LEFT(1, l) * RIGHT(m, 0) * grad_nmat[4*k+1] + \
                                            LEFT(0, l) * RIGHT(m, 1) * grad_nmat[4*k+2] + \
                                            LEFT(1, l) * RIGHT(m, 1) * grad_nmat[4*k+3];
                    }
                }
                k++;
            }
        }
    }
}