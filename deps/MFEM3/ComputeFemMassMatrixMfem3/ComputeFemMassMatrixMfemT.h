#include "../Common.h"

namespace MFEM{
    void ComputeFemMassMatrixMfemT_forward(int64*indices, double *vv, const double *rho){
        int elem_ndof = mmesh3.elem_ndof;
        int nz = 0;
        for(int i = 0; i < mmesh3.nelem; i++){
            NNFEM_Element3 * elem = mmesh3.elements[i];

            for (int p = 0; p < elem_ndof; p ++)
                for (int q = 0; q< elem_ndof; q++){

                    double s = 0.0;
                    for (int k = 0; k<elem->ngauss; k++){
                        s += elem->h(p, k) * elem->h(q, k) * elem->w[k] * rho[k];
                    }
                    
                    indices[2*nz] = elem->dof[p];
                    indices[2*nz+1] = elem->dof[q];
                    vv[nz] = s;
                    nz ++;
            }

            rho += elem->ngauss;

        }
    }
}
