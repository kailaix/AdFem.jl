#include "../Common.h"

typedef  long long int64;
extern "C" void ComputeInteractionMatrixMfem(int64 *ii, int64 *jj, double *vv){
    int s = 0;
    for(int i = 0; i<mmesh.nelem; i++){
        NNFEM_Element * elem = mmesh.elements[i];
        for (int j = 0; j< elem->ngauss; j++)
            for (int k = 0; k < 3; k++){
                ii[s] = i + 1;
                jj[s] = elem->node[k] + 1;
                vv[s] = (elem->hx(k, j) + elem->hy(k, j)) * elem->w[j];
                s++;
            }
    }
}
