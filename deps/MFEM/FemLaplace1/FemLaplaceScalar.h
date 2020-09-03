#include "../Common.h"
namespace MFEM{
  void FemLaplaceScalar_forward(int64 *indices, double *vv, const double *kappa){
    int s = 0;
    int nz = 0;
    for(int i = 0; i<mmesh.nelem; i++){
        NNFEM_Element * elem = mmesh.elements[i];
        Eigen::MatrixXd D(3, 2);
        for (int k = 0; k<elem->ngauss; k++){
          D << elem->hx(0, k), elem->hy(0, k),
             elem->hx(1, k), elem->hy(1, k),
             elem->hx(2, k), elem->hy(2, k);
          Eigen::MatrixXd N = D * D.transpose() * kappa[s++] * elem->w[k];
          for (int p = 0; p < 3; p ++)
            for (int q = 0; q<3; q++){
              indices[2*nz] = elem->node[p];
              indices[2*nz+1] = elem->node[q];
              vv[nz] = N(p, q);
              nz ++;
            }
        }
        
    }
  }

  void FemLaplaceScalar_backward(int64 *ii, int64 *jj, double *vv, const double *kappa){
    
  }

}
