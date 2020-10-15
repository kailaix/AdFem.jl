#include "../Common.h"
extern "C" void BDMSkewSymmetricMatrixMfem(long long*ii, long long*jj, double *vv){
    if (mmesh.degree!=-1){
        printf("ERROR (BDMSkewSymmetricMatrixMfem): expected BDM1 finite element type\n");
        return;
    }
    int elem_ndof = mmesh.elem_ndof;    
    int k = 0;
    for(int i = 0; i<mmesh.nelem; i++){
        NNFEM_Element * elem = mmesh.elements[i];
        for (int j = 0; j< elem->ngauss; j++){
            for (int r = 0; r<elem_ndof; r++){
                ii[k] = i + 1;
                jj[k] = elem->dof[r] + 1;
                vv[k] = elem->w[j] * elem->BDMy(r, j);
                k++;
                ii[k] = i + 1;
                jj[k] = elem->dof[r] + 1 + mmesh.ndof;
                vv[k] = -elem->w[j] * elem->BDMx(r, j);
                k++;
            }
        }
    }
}